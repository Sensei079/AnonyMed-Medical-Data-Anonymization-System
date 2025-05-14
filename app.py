import pandas as pd
import numpy as np
import os
import re
import tempfile
import hashlib
import uuid
import gradio as gr
from datetime import datetime
from typing import Dict, List, Union, Tuple, Optional
import logging
import warnings
from cryptography.fernet import Fernet
import csv
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Set up storage directories
STORAGE_DIR = './anonymed_data'
os.makedirs(STORAGE_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=os.path.join(STORAGE_DIR, 'anonymization_system.log')
)
logger = logging.getLogger('anonymed')

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Constants for anonymization
HASH_SALT = 'AnonyMed2025' 
AGE_BINS = [0, 18, 30, 40, 50, 60, 70, 80, 120]
AGE_LABELS = ['0-18', '19-30', '31-40', '41-50', '51-60', '61-70', '71-80', '80+']
DEMOGRAPHIC_COLS = ['Gender', 'Blood']
K_ANONYMITY_DEFAULT = 5  # Default k-anonymity parameter

# Database files
DATABASE_FILE = os.path.join(STORAGE_DIR, 'patient_database.csv')
MAPPING_FILE = os.path.join(STORAGE_DIR, 'id_mapping.encrypted')
KEY_FILE = os.path.join(STORAGE_DIR, 'encryption.key')

# Generate an encryption key or load existing one
if os.path.exists(KEY_FILE):
    with open(KEY_FILE, 'rb') as key_file:
        ENCRYPTION_KEY = key_file.read()
else:
    ENCRYPTION_KEY = Fernet.generate_key()
    with open(KEY_FILE, 'wb') as key_file:
        key_file.write(ENCRYPTION_KEY)

cipher_suite = Fernet(ENCRYPTION_KEY)

# Try to load spaCy model
try:
    import spacy
    nlp = spacy.load("en_core_web_md")
    logger.info("Loaded spaCy model en_core_web_md")
except OSError:
    # If the larger model isn't available, use the small one
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("Loaded spaCy model en_core_web_sm")
    except:
        # If spaCy is not available, add fallback
        logger.warning("SpaCy not available. Using simplified NLP.")
        class SimplifiedNLP:
            def __call__(self, text):
                return SimpleDoc(text)

        class SimpleDoc:
            def __init__(self, text):
                self.text = text
                self.ents = []

        nlp = SimplifiedNLP()

# Try to load Universal Sentence Encoder for text similarity
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    use_model_available = True
    logger.info("Loaded Universal Sentence Encoder")
except Exception as e:
    use_model_available = False
    logger.warning(f"Could not load Universal Sentence Encoder: {e}")

# Class to handle the ID mapping system
class IDMapper:
    def __init__(self, mapping_file=MAPPING_FILE):
        self.mapping_file = mapping_file
        self.id_map = {}
        self.load_mapping()

    def load_mapping(self):
        if os.path.exists(self.mapping_file):
            try:
                with open(self.mapping_file, 'rb') as f:
                    encrypted_data = f.read()
                decrypted_data = cipher_suite.decrypt(encrypted_data)
                self.id_map = json.loads(decrypted_data.decode())
                logger.info(f"Loaded {len(self.id_map)} ID mappings")
            except Exception as e:
                logger.error(f"Error loading ID mappings: {e}")
                self.id_map = {}

    def save_mapping(self):
        try:
            data = json.dumps(self.id_map).encode()
            encrypted_data = cipher_suite.encrypt(data)
            with open(self.mapping_file, 'wb') as f:
                f.write(encrypted_data)
            logger.info(f"Saved {len(self.id_map)} ID mappings")
        except Exception as e:
            logger.error(f"Error saving ID mappings: {e}")

    def get_pseudonym(self, identifier):
        """Generate a consistent pseudonym for a given identifier"""
        if identifier not in self.id_map:
            # Create a UUID based on a hash of the identifier with salt
            hash_base = hashlib.sha256((identifier + HASH_SALT).encode()).hexdigest()
            pseudonym = str(uuid.UUID(hash_base[:32]))
            # Store only the first 8 chars of the pseudonym as the "friendly" ID
            friendly_id = f"Patient_{pseudonym[:8]}"
            self.id_map[identifier] = {"uuid": pseudonym, "friendly_id": friendly_id}
            self.save_mapping()
        return self.id_map[identifier]["friendly_id"]

# Initialize the ID mapper
id_mapper = IDMapper()

# Load previous data if it exists
def load_database() -> pd.DataFrame:
    """Load the anonymized database"""
    if os.path.exists(DATABASE_FILE):
        try:
            return pd.read_csv(DATABASE_FILE)
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

# Save data to CSV
def save_database(df: pd.DataFrame):
    """Save the anonymized database"""
    if not df.empty:
        try:
            df.to_csv(DATABASE_FILE, index=False)
            logger.info(f"Saved database with {len(df)} records")
        except Exception as e:
            logger.error(f"Error saving database: {e}")

# Patient data anonymization functions
class Anonymizer:
    @staticmethod
    def anonymize_name(name: str) -> str:
        """Generate consistent pseudonym for name"""
        return id_mapper.get_pseudonym(name.lower())

    @staticmethod
    def anonymize_age(age: int) -> str:
        """Convert age to age bracket"""
        try:
            age = int(age)
            age_bracket = pd.cut([age], bins=AGE_BINS, labels=AGE_LABELS)[0]
            return str(age_bracket)
        except:
            return "Unknown"

    @staticmethod
    def anonymize_address(address: str) -> str:
        """Extract only city/region from address and generalize"""
        # First try to extract the city using NLP
        doc = nlp(address)
        cities = [ent.text for ent in doc.ents if hasattr(ent, 'label_') and ent.label_ == "GPE"]

        if cities:
            # If we found cities, use the first one
            city_hash = hashlib.md5((cities[0].lower() + HASH_SALT).encode()).hexdigest()
            return f"Region_{city_hash[:5]}"
        else:
            # If no city found, hash the entire address but return only a short prefix
            addr_hash = hashlib.md5((address.lower() + HASH_SALT).encode()).hexdigest()
            return f"Area_{addr_hash[:5]}"

    @staticmethod
    def anonymize_medical_condition(condition: str) -> str:
        """Preserve medical condition category while obscuring specifics"""
        # Common medical condition categories
        categories = {
            'heart': ['heart', 'cardiac', 'cardiovascular', 'coronary', 'arrhythmia', 'hypertension'],
            'respiratory': ['lung', 'asthma', 'copd', 'respiratory', 'pneumonia', 'bronchitis'],
            'diabetes': ['diabetes', 'insulin', 'glucose', 'glycemic'],
            'cancer': ['cancer', 'tumor', 'oncology', 'carcinoma', 'leukemia', 'lymphoma'],
            'mental_health': ['depression', 'anxiety', 'schizophrenia', 'bipolar', 'mental', 'psychiatric'],
            'neurological': ['alzheimer', 'parkinson', 'epilepsy', 'seizure', 'neurological', 'brain'],
            'musculoskeletal': ['arthritis', 'osteoporosis', 'fracture', 'joint', 'bone', 'muscle'],
            'gastrointestinal': ['gastric', 'ulcer', 'ibs', 'crohn', 'colitis', 'liver', 'hepatitis'],
            'immune': ['allergy', 'autoimmune', 'lupus', 'hiv', 'aids', 'immune'],
            'infection': ['infection', 'bacterial', 'viral', 'fungal'],
            'other': []  # Default category
        }

        condition_lower = condition.lower()

        # Check which category the condition falls into
        for category, keywords in categories.items():
            if any(keyword in condition_lower for keyword in keywords):
                # Create a hash to ensure consistency but prevent exact identification
                cond_hash = hashlib.md5((condition.lower() + HASH_SALT).encode()).hexdigest()[:4]
                return f"{category.capitalize()}_{cond_hash}"

        # If no specific category matched
        cond_hash = hashlib.md5((condition.lower() + HASH_SALT).encode()).hexdigest()[:4]
        return f"Condition_{cond_hash}"

    @staticmethod
    def detect_phi(text: str) -> List[str]:
        """Detect potential PHI (Protected Health Information) in text"""
        phi_entities = []
        doc = nlp(text)

        # Check for named entities that might be PHI
        for ent in doc.ents:
            if hasattr(ent, 'label_') and ent.label_ in ["PERSON", "GPE", "LOC", "ORG", "DATE"]:
                phi_entities.append(ent.text)

        # Check for phone numbers
        phone_pattern = re.compile(r'\b(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b')
        phone_matches = phone_pattern.findall(text)
        phi_entities.extend(phone_matches)

        # Check for emails
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        email_matches = email_pattern.findall(text)
        phi_entities.extend(email_matches)

        # Check for SSNs
        ssn_pattern = re.compile(r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b')
        ssn_matches = ssn_pattern.findall(text)
        phi_entities.extend(ssn_matches)

        return list(set(phi_entities))  

    @staticmethod
    def risk_score(df: pd.DataFrame) -> float:
        """Calculate re-identification risk score"""
        # Simple k-anonymity implementation
        if df.empty:
            return 0.0

        # Count occurrences of each unique combination of quasi-identifiers
        quasi_identifiers = ['Anonymized Age', 'Gender', 'Blood', 'Anonymized Address']
        available_qi = [col for col in quasi_identifiers if col in df.columns]

        if not available_qi:
            return 0.0

        # Count how many records have each combination of quasi-identifiers
        counts = df.groupby(available_qi).size().reset_index(name='count')

        # Find the smallest group size (k value)
        k = counts['count'].min() if not counts.empty else 0

        # Calculate risk as 1/k (capped at 1.0)
        risk = 1.0 / k if k > 0 else 1.0
        return min(risk, 1.0)

# Function to anonymize patient data
def anonymize_data(
    name: str,
    age: str,
    address: str,
    gender: str,
    blood: str,
    medical_condition: str,
    extra_notes: str = ""
) -> Union[pd.DataFrame, str]:
    """Anonymize patient data using advanced techniques"""
    try:
        # Input validation
        if not name or not address:
            raise ValueError("Name and address cannot be empty.")

        try:
            age_val = int(age)
            if age_val <= 0 or age_val > 120:
                raise ValueError("Age must be a positive integer between 1 and 120.")
        except ValueError:
            raise ValueError("Age must be a valid number.")

        # Check for PHI in notes
        phi_found = []
        if extra_notes:
            phi_found = Anonymizer.detect_phi(extra_notes)

        # Perform anonymization
        anonymized_name = Anonymizer.anonymize_name(name)
        anonymized_age = Anonymizer.anonymize_age(age_val)
        anonymized_address = Anonymizer.anonymize_address(address)
        anonymized_condition = Anonymizer.anonymize_medical_condition(medical_condition)

        # Create anonymized notes if provided
        anonymized_notes = ""
        if extra_notes:
            # Replace any detected PHI
            anonymized_notes = extra_notes
            for phi in phi_found:
                anonymized_notes = anonymized_notes.replace(phi, "[REDACTED]")

        # Calculate creation timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create a unique record ID
        record_id = str(uuid.uuid4())

        # Create the anonymized record
        anonymized_record = {
            'Record ID': record_id,
            'Anonymized Name': anonymized_name,
            'Anonymized Age': anonymized_age,
            'Anonymized Address': anonymized_address,
            'Gender': gender,
            'Blood': blood,
            'Anonymized Condition': anonymized_condition,
            'Anonymized Notes': anonymized_notes,
            'Timestamp': timestamp,
            # Encrypted original data for authorized access
            'Original Data': cipher_suite.encrypt(
                json.dumps({
                    'name': name,
                    'age': age_val,
                    'address': address,
                    'condition': medical_condition,
                    'notes': extra_notes
                }).encode()
            ).decode()
        }

        return pd.DataFrame([anonymized_record])

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return str(e)
    except Exception as e:
        logger.error(f"Anonymization error: {e}")
        return f"An error occurred during anonymization: {str(e)}"

# Function to save anonymized data
def save_anonymized_data(
    name: str,
    age: str,
    address: str,
    gender: str,
    blood: str,
    medical_condition: str,
    extra_notes: str = ""
) -> Union[pd.DataFrame, str]:
    """Process and save anonymized patient data"""
    # Anonymize the data
    result = anonymize_data(name, age, address, gender, blood, medical_condition, extra_notes)

    if not isinstance(result, pd.DataFrame):
        return result  

    # Load existing database
    database = load_database()

    # Append new record
    if database.empty:
        database = result
    else:
        database = pd.concat([database, result], ignore_index=True)

    # Save updated database
    save_database(database)

    # Return only the anonymized columns for display
    display_cols = [col for col in database.columns if not col.startswith('Original') and col != 'Original Data']
    return database[display_cols].tail(10)  # Return last 10 entries

# Function to export anonymized data to CSV
def export_anonymized_data() -> str:
    """Export anonymized data to a timestamped CSV file"""
    database = load_database()

    if database.empty:
        return "No data available for export"

    # Get only the anonymized columns
    export_cols = [col for col in database.columns if not col.startswith('Original') and col != 'Original Data']
    export_df = database[export_cols]

    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(STORAGE_DIR, f"anonymized_data_{timestamp}.csv")

    # Export to CSV
    try:
        export_df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(export_df)} records to {output_path}")

        # For Colab, try to download the file
        try:
            files.download(output_path)
        except Exception as e:
            logger.warning(f"Could not trigger browser download: {e}")

        return f"Data successfully exported to: {output_path}"
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        return f"Error exporting data: {str(e)}"

# Function to check k-anonymity status
def check_k_anonymity(k: int = K_ANONYMITY_DEFAULT) -> Dict:
    """Check if the dataset meets k-anonymity requirements"""
    database = load_database()

    if database.empty:
        return {
            "status": "No data available",
            "violations": 0,
            "total_groups": 0,
            "risk_score": 0.0
        }

    # Use demographic columns as quasi-identifiers
    quasi_identifiers = ['Anonymized Age', 'Gender', 'Blood', 'Anonymized Address']
    available_qi = [col for col in quasi_identifiers if col in database.columns]

    if not available_qi:
        return {
            "status": "No quasi-identifiers available",
            "violations": 0,
            "total_groups": 0,
            "risk_score": 0.0
        }

    # Group by quasi-identifiers and count occurrences
    counts = database.groupby(available_qi).size().reset_index(name='count')
    violations = counts[counts['count'] < k]

    risk_score = Anonymizer.risk_score(database)

    return {
        "status": "Dataset meets k-anonymity" if violations.empty else f"Dataset violates k-anonymity ({len(violations)} groups)",
        "violations": len(violations),
        "total_groups": len(counts),
        "risk_score": risk_score
    }

# Function to visualize anonymized data
def visualize_anonymization():
    """Generate visualizations of anonymized data"""
    database = load_database()

    if database.empty:
        return {
            "status": "No data available for visualization",
            "age_distribution": None,
            "gender_distribution": None,
            "risk_over_time": None
        }

    # Create plots
    plots = []

    try:
        # 1. Age Distribution
        if 'Anonymized Age' in database.columns:
            plt.figure(figsize=(10, 6))
            age_counts = database['Anonymized Age'].value_counts().sort_index()
            sns.barplot(x=age_counts.index, y=age_counts.values)
            plt.title('Distribution of Anonymized Age Groups')
            plt.xlabel('Age Group')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            age_dist_path = os.path.join(STORAGE_DIR, 'age_distribution.png')
            plt.tight_layout()
            plt.savefig(age_dist_path)
            plt.close()
            plots.append((age_dist_path, "Age Distribution"))

        # 2. Gender Distribution
        if 'Gender' in database.columns:
            plt.figure(figsize=(8, 6))
            gender_counts = database['Gender'].value_counts()
            sns.barplot(x=gender_counts.index, y=gender_counts.values)
            plt.title('Distribution of Gender')
            plt.xlabel('Gender')
            plt.ylabel('Count')
            gender_dist_path = os.path.join(STORAGE_DIR, 'gender_distribution.png')
            plt.tight_layout()
            plt.savefig(gender_dist_path)
            plt.close()
            plots.append((gender_dist_path, "Gender Distribution"))

        # 3. Medical Condition Categories
        if 'Anonymized Condition' in database.columns:
            plt.figure(figsize=(12, 6))
            condition_categories = database['Anonymized Condition'].apply(lambda x: x.split('_')[0] if '_' in x else x)
            condition_counts = condition_categories.value_counts().head(10)  # Top 10 categories
            sns.barplot(x=condition_counts.index, y=condition_counts.values)
            plt.title('Top Medical Condition Categories')
            plt.xlabel('Category')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            condition_dist_path = os.path.join(STORAGE_DIR, 'condition_distribution.png')
            plt.tight_layout()
            plt.savefig(condition_dist_path)
            plt.close()
            plots.append((condition_dist_path, "Condition Distribution"))

    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        return {
            "status": f"Error generating visualizations: {str(e)}",
            "error": str(e)
        }

    return plots if plots else {
        "status": "No visualizations could be generated"
    }

# Function to authorize access to original data (typically would require authentication)
def authorize_access(record_id: str, purpose: str) -> Union[pd.DataFrame, str]:
    """Simulated authorized access to original data with purpose logging"""
    database = load_database()

    if database.empty:
        return "No data available"

    # Find the record
    record = database[database['Record ID'] == record_id]

    if record.empty:
        return f"Record with ID {record_id} not found"

    try:
        # Log access attempt
        logger.info(f"Data access requested for record {record_id} - Purpose: {purpose}")

        # Decrypt original data
        encrypted_data = record['Original Data'].iloc[0]
        decrypted_bytes = cipher_suite.decrypt(encrypted_data.encode())
        original_data = json.loads(decrypted_bytes.decode())

        # Return as DataFrame for display
        return pd.DataFrame([{
            'Record ID': record_id,
            'Name': original_data['name'],
            'Age': original_data['age'],
            'Address': original_data['address'],
            'Medical Condition': original_data['condition'],
            'Notes': original_data['notes'],
            'Access Purpose': purpose,
            'Access Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }])

    except Exception as e:
        logger.error(f"Error decrypting data: {e}")
        return f"Error accessing data: {str(e)}"

# Function to perform cohort analysis on anonymized data
def perform_cohort_analysis():
    """Perform basic cohort analysis on anonymized data"""
    database = load_database()

    if database.empty:
        return "No data available for analysis"

    analysis_results = {}

    try:
        # Count by condition and age group
        if 'Anonymized Condition' in database.columns and 'Anonymized Age' in database.columns:
            # Extract condition categories
            database['Condition Category'] = database['Anonymized Condition'].apply(
                lambda x: x.split('_')[0] if '_' in x else x
            )

            # Create pivot table
            pivot = pd.pivot_table(
                database,
                values='Record ID',
                index='Condition Category',
                columns='Anonymized Age',
                aggfunc='count',
                fill_value=0
            )

            # Convert to percentage for better visualization
            total_by_condition = pivot.sum(axis=1)
            # Handle division by zero
            percentage_pivot = pivot.div(total_by_condition.replace(0, np.nan), axis=0) * 100

            # Plot heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(percentage_pivot, annot=True, cmap="YlGnBu", fmt=".1f")
            plt.title('Condition Categories by Age Group (%)')
            plt.xlabel('Age Group')
            plt.ylabel('Condition Category')
            plt.tight_layout()
            heatmap_path = os.path.join(STORAGE_DIR, 'condition_age_heatmap.png')
            plt.savefig(heatmap_path)
            plt.close()

            analysis_results['condition_age_heatmap'] = heatmap_path
            analysis_results['pivot_table'] = pivot.to_dict()
        else:
            analysis_results['message'] = "Missing required columns for cohort analysis"

        # Try to find patterns using clustering
        if len(database) > 5:  
            # Prepare data for clustering
            if all(col in database.columns for col in ['Gender', 'Blood', 'Anonymized Age']):
                cat_features = pd.get_dummies(database[['Gender', 'Blood', 'Anonymized Age']])

                # Standardize features
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(cat_features)

                # Determine optimal number of clusters (simplified)
                k_values = range(2, min(6, len(cat_features) // 2))
                inertias = []

                for k in k_values:
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    kmeans.fit(scaled_features)
                    inertias.append(kmeans.inertia_)

                # Plot elbow curve
                plt.figure(figsize=(10, 6))
                plt.plot(k_values, inertias, 'o-')
                plt.xlabel('Number of Clusters (k)')
                plt.ylabel('Inertia')
                plt.title('Elbow Method for Optimal k')
                plt.grid(True)
                elbow_path = os.path.join(STORAGE_DIR, 'elbow_curve.png')
                plt.savefig(elbow_path)
                plt.close()

                # Use best k from elbow method (simplified)
                best_k = min(3, len(k_values))  
                kmeans = KMeans(n_clusters=best_k, random_state=42)
                clusters = kmeans.fit_predict(scaled_features)

                # Add cluster to database for visualization
                db_with_clusters = database.copy()
                db_with_clusters['Cluster'] = clusters

                # Visualize clusters using t-SNE
                if len(database) >= 5: 
                    tsne = TSNE(n_components=2, random_state=42)
                    tsne_results = tsne.fit_transform(scaled_features)

                    plt.figure(figsize=(10, 8))
                    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=clusters, cmap='viridis', alpha=0.8)
                    plt.title('t-SNE Visualization of Patient Clusters')
                    plt.colorbar(label='Cluster')
                    plt.xlabel('t-SNE dimension 1')
                    plt.ylabel('t-SNE dimension 2')
                    tsne_path = os.path.join(STORAGE_DIR, 'tsne_clusters.png')
                    plt.savefig(tsne_path)
                    plt.close()

                    analysis_results['elbow_curve'] = elbow_path
                    analysis_results['tsne_clusters'] = tsne_path

                # Analyze cluster characteristics
                cluster_profiles = {}
                for i in range(best_k):
                    cluster_data = db_with_clusters[db_with_clusters['Cluster'] == i]
                    profile = {
                        'size': len(cluster_data),
                        'percent': round(len(cluster_data) / len(database) * 100, 1),
                    }

                    # Check for 'Condition Category' column before using it
                    if 'Condition Category' in cluster_data.columns:
                        profile['top_conditions'] = cluster_data['Condition Category'].value_counts().head(3).to_dict()

                    # Add other properties if columns exist
                    if 'Anonymized Age' in cluster_data.columns:
                        profile['age_groups'] = cluster_data['Anonymized Age'].value_counts().to_dict()

                    if 'Gender' in cluster_data.columns:
                        profile['gender_ratio'] = cluster_data['Gender'].value_counts().to_dict()

                    cluster_profiles[f'Cluster_{i}'] = profile

                analysis_results['cluster_profiles'] = cluster_profiles
            else:
                analysis_results['clustering_error'] = "Missing required columns for clustering analysis"
        else:
            analysis_results['clustering_error'] = "Not enough data for clustering analysis"

    except Exception as e:
        logger.error(f"Error in cohort analysis: {e}")
        analysis_results['error'] = str(e)

    return analysis_results

# Function to generate a privacy dashboard
def generate_privacy_dashboard():
    """Generate a privacy dashboard for the anonymized dataset"""
    database = load_database()

    if database.empty:
        return "No data available for dashboard"

    dashboard_data = {}

    try:
        # Calculate basic metrics
        total_records = len(database)
        dashboard_data['total_records'] = total_records

        # Calculate k-anonymity status
        k_anonymity_data = check_k_anonymity()
        dashboard_data['k_anonymity'] = k_anonymity_data

        # Calculate risk distribution
        if all(col in database.columns for col in ['Anonymized Age', 'Gender', 'Blood', 'Anonymized Address']):
            # Create quasi-identifier combinations
            quasi_identifiers = ['Anonymized Age', 'Gender', 'Blood', 'Anonymized Address']
            available_qi = [col for col in quasi_identifiers if col in database.columns]

            # Group by quasi-identifiers and count
            counts = database.groupby(available_qi).size().reset_index(name='count')

            # Create a risk metric (1/count for each group)
            counts['risk'] = 1 / counts['count']

            # Plot risk distribution
            plt.figure(figsize=(10, 6))
            plt.hist(counts['risk'], bins=10, alpha=0.7)
            plt.axvline(x=0.2, color='red', linestyle='--', label='High Risk Threshold')
            plt.title('Re-identification Risk Distribution')
            plt.xlabel('Risk Score (1/k)')
            plt.ylabel('Number of Groups')
            plt.legend()
            plt.grid(True, alpha=0.3)
            risk_path = os.path.join(STORAGE_DIR, 'risk_distribution.png')
            plt.tight_layout()
            plt.savefig(risk_path)
            plt.close()

            dashboard_data['risk_distribution'] = risk_path
            dashboard_data['high_risk_groups'] = len(counts[counts['risk'] > 0.2])

        # Generate condition distribution chart
        if 'Anonymized Condition' in database.columns:
            # Extract condition categories
            database['Condition Category'] = database['Anonymized Condition'].apply(
                lambda x: x.split('_')[0] if '_' in x else x
            )

            # Get top condition categories
            top_conditions = database['Condition Category'].value_counts().head(8)

            plt.figure(figsize=(10, 6))
            sns.barplot(x=top_conditions.index, y=top_conditions.values)
            plt.title('Top Medical Condition Categories')
            plt.xlabel('Category')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            condition_path = os.path.join(STORAGE_DIR, 'condition_dashboard.png')
            plt.tight_layout()
            plt.savefig(condition_path)
            plt.close()

            dashboard_data['condition_distribution'] = condition_path

        # Create privacy metrics summary
        dashboard_data['privacy_metrics'] = {
            'unique_ids': len(database['Anonymized Name'].unique()),
            'unique_records': len(database.drop_duplicates()),
            'risk_score': k_anonymity_data['risk_score'],
            'safety_status': 'Safe' if k_anonymity_data['risk_score'] < 0.2 else 'Potentially Unsafe'
        }

        # Generate HTML report
        html_report = f"""
        <h1>AnonyMed Privacy Dashboard</h1>
        <h2>Dataset Overview</h2>
        <p>Total Records: {total_records}</p>
        <p>Unique Patient IDs: {dashboard_data['privacy_metrics']['unique_ids']}</p>
        <p>Risk Score: {dashboard_data['privacy_metrics']['risk_score']:.4f}</p>
        <p>Safety Status: {dashboard_data['privacy_metrics']['safety_status']}</p>

        <h2>K-anonymity Status</h2>
        <p>{k_anonymity_data['status']}</p>
        <p>Total Groups: {k_anonymity_data['total_groups']}</p>
        <p>Violations: {k_anonymity_data['violations']}</p>
        """

        dashboard_path = os.path.join(STORAGE_DIR, 'privacy_dashboard.html')
        with open(dashboard_path, 'w') as f:
            f.write(html_report)

        dashboard_data['dashboard_path'] = dashboard_path

    except Exception as e:
        logger.error(f"Error generating privacy dashboard: {e}")
        dashboard_data['error'] = str(e)

    return dashboard_data

# Fixes for the export_anonymized_data function
def export_anonymized_data_for_download():
    """Prepare anonymized data for direct download via Gradio"""
    database = load_database()

    if database.empty:
        return None, "No data available for download", gr.update(visible=False)

    # Get only the anonymized columns
    export_cols = [col for col in database.columns if not col.startswith('Original') and col != 'Original Data']
    export_df = database[export_cols]

    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"anonymized_data_{timestamp}.csv"
    
    try:
        # Create a temporary file
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, filename)
        
        # Save to CSV
        export_df.to_csv(temp_path, index=False)
        logger.info(f"Exported {len(export_df)} records to {temp_path}")
        
        success_message = f"✅ Successfully prepared {len(export_df)} records for download"
        return temp_path, success_message, gr.update(visible=True)
    
    except Exception as e:
        logger.error(f"Error preparing data for download: {e}")
        return None, f"❌ Error preparing data: {str(e)}", gr.update(visible=False)

# Fix for the visualize_anonymization function
def visualize_anonymization():
    """Generate visualizations of anonymized data"""
    database = load_database()

    if database.empty:
        return "No data available for visualization"

    # Create plots
    plots = []

    try:
        # Make sure storage directory exists
        os.makedirs(STORAGE_DIR, exist_ok=True)

        # 1. Age Distribution
        if 'Anonymized Age' in database.columns:
            plt.figure(figsize=(10, 6))
            age_counts = database['Anonymized Age'].value_counts().sort_index()
            sns.barplot(x=age_counts.index, y=age_counts.values)
            plt.title('Distribution of Anonymized Age Groups')
            plt.xlabel('Age Group')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            age_dist_path = os.path.join(STORAGE_DIR, 'age_distribution.png')
            plt.tight_layout()
            plt.savefig(age_dist_path)
            plt.close()
            plots.append((age_dist_path, "Age Distribution"))

            # Display the plot directly in Colab
            img = plt.imread(age_dist_path)
            plt.figure(figsize=(10, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.title('Age Distribution')
            plt.close()

        # 2. Gender Distribution
        if 'Gender' in database.columns:
            plt.figure(figsize=(8, 6))
            gender_counts = database['Gender'].value_counts()
            sns.barplot(x=gender_counts.index, y=gender_counts.values)
            plt.title('Distribution of Gender')
            plt.xlabel('Gender')
            plt.ylabel('Count')
            gender_dist_path = os.path.join(STORAGE_DIR, 'gender_distribution.png')
            plt.tight_layout()
            plt.savefig(gender_dist_path)
            plt.close()
            plots.append((gender_dist_path, "Gender Distribution"))

            # Display the plot directly in Colab
            img = plt.imread(gender_dist_path)
            plt.figure(figsize=(8, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.title('Gender Distribution')
            plt.close()

        # 3. Medical Condition Categories
        if 'Anonymized Condition' in database.columns:
            plt.figure(figsize=(12, 6))
            # Extract condition categories
            condition_categories = database['Anonymized Condition'].apply(
                lambda x: x.split('_')[0] if '_' in x else x
            )
            condition_counts = condition_categories.value_counts().head(10)  
            
            if not condition_counts.empty:
                sns.barplot(x=condition_counts.index, y=condition_counts.values)
                plt.title('Top Medical Condition Categories')
                plt.xlabel('Category')
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                condition_dist_path = os.path.join(STORAGE_DIR, 'condition_distribution.png')
                plt.tight_layout()
                plt.savefig(condition_dist_path)
                plt.close()
                plots.append((condition_dist_path, "Condition Distribution"))

                # Display the plot directly in Colab
                img = plt.imread(condition_dist_path)
                plt.figure(figsize=(12, 6))
                plt.imshow(img)
                plt.axis('off')
                plt.title('Condition Distribution')
                plt.close()

        # 4. Blood Type Distribution
        if 'Blood' in database.columns:
            plt.figure(figsize=(8, 6))
            blood_counts = database['Blood'].value_counts()
            sns.barplot(x=blood_counts.index, y=blood_counts.values)
            plt.title('Distribution of Blood Types')
            plt.xlabel('Blood Type')
            plt.ylabel('Count')
            blood_dist_path = os.path.join(STORAGE_DIR, 'blood_distribution.png')
            plt.tight_layout()
            plt.savefig(blood_dist_path)
            plt.close()
            plots.append((blood_dist_path, "Blood Type Distribution"))

            # Display the plot directly in Colab
            img = plt.imread(blood_dist_path)
            plt.figure(figsize=(8, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.title('Blood Type Distribution')
            plt.close()

    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        return f"Error generating visualizations: {str(e)}"

    return plots if plots else "No visualizations could be generated"

# Fix for the perform_cohort_analysis function
def perform_cohort_analysis():
    """Perform basic cohort analysis on anonymized data"""
    database = load_database()

    if database.empty:
        return "No data available for analysis"

    analysis_results = {}

    try:
        # Make sure the storage directory exists
        os.makedirs(STORAGE_DIR, exist_ok=True)

        # Count by condition and age group
        if 'Anonymized Condition' in database.columns and 'Anonymized Age' in database.columns:
            # Extract condition categories
            database['Condition Category'] = database['Anonymized Condition'].apply(
                lambda x: x.split('_')[0] if '_' in x else x
            )

            # Create pivot table
            pivot = pd.pivot_table(
                database,
                values='Record ID',
                index='Condition Category',
                columns='Anonymized Age',
                aggfunc='count',
                fill_value=0
            )

            # Handle empty pivot table
            if pivot.empty:
                analysis_results['message'] = "Not enough data for pivot table analysis"
            else:
                # Convert to percentage for better visualization
                total_by_condition = pivot.sum(axis=1)
                # Handle division by zero
                total_replaced = total_by_condition.replace(0, np.nan)
                percentage_pivot = pivot.div(total_replaced, axis=0) * 100

                # Plot heatmap
                plt.figure(figsize=(12, 8))
                sns.heatmap(percentage_pivot, annot=True, cmap="YlGnBu", fmt=".1f")
                plt.title('Condition Categories by Age Group (%)')
                plt.xlabel('Age Group')
                plt.ylabel('Condition Category')
                plt.tight_layout()
                heatmap_path = os.path.join(STORAGE_DIR, 'condition_age_heatmap.png')
                plt.savefig(heatmap_path)
                plt.close()

                analysis_results['condition_age_heatmap'] = heatmap_path
                analysis_results['pivot_table'] = pivot.to_dict()

                # Show the heatmap directly in Colab
                img = plt.imread(heatmap_path)
                plt.figure(figsize=(12, 8))
                plt.imshow(img)
                plt.axis('off')
                plt.title('Condition Categories by Age Group (%)')
                plt.close()
        else:
            analysis_results['message'] = "Missing required columns for cohort analysis"

        # Try to find patterns using clustering
        if len(database) > 5:  
            # Prepare data for clustering
            if all(col in database.columns for col in ['Gender', 'Blood']):
                # Create categorical features
                cat_cols = ['Gender', 'Blood']
                if 'Anonymized Age' in database.columns:
                    cat_cols.append('Anonymized Age')

                # Handle missing values
                data_for_cluster = database[cat_cols].fillna('Unknown')

                # Convert to numeric using one-hot encoding
                cat_features = pd.get_dummies(data_for_cluster)

                if cat_features.empty or cat_features.shape[1] == 0:
                    analysis_results['clustering_error'] = "Not enough categorical features for clustering"
                else:
                    # Standardize features
                    scaler = StandardScaler()
                    scaled_features = scaler.fit_transform(cat_features)

                    # Determine optimal number of clusters (simplified)
                    max_clusters = min(6, len(database) // 2, 10)  
                    if max_clusters < 2:
                        max_clusters = 2

                    k_values = range(2, max_clusters)
                    inertias = []

                    for k in k_values:
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        kmeans.fit(scaled_features)
                        inertias.append(kmeans.inertia_)

                    # Plot elbow curve
                    plt.figure(figsize=(10, 6))
                    plt.plot(k_values, inertias, 'o-')
                    plt.xlabel('Number of Clusters (k)')
                    plt.ylabel('Inertia')
                    plt.title('Elbow Method for Optimal k')
                    plt.grid(True)
                    elbow_path = os.path.join(STORAGE_DIR, 'elbow_curve.png')
                    plt.savefig(elbow_path)
                    plt.close()

                    # Use best k from elbow method (simplified)
                    best_k = 2 
                    if len(k_values) > 0:
                        best_k = k_values[0] + 1 
                        if best_k > max_clusters - 1:
                            best_k = max_clusters - 1

                    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(scaled_features)

                    # Add cluster to database for visualization
                    db_with_clusters = database.copy()
                    db_with_clusters['Cluster'] = clusters

                    # Visualize clusters using t-SNE if we have enough data
                    if len(database) >= 5:  
                        perplexity = max(2, len(database) // 3)
                        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)

                        try:
                            tsne_results = tsne.fit_transform(scaled_features)

                            plt.figure(figsize=(10, 8))
                            scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1],
                                      c=clusters, cmap='viridis', alpha=0.8)
                            plt.title('t-SNE Visualization of Patient Clusters')
                            plt.colorbar(scatter, label='Cluster')
                            plt.xlabel('t-SNE dimension 1')
                            plt.ylabel('t-SNE dimension 2')
                            tsne_path = os.path.join(STORAGE_DIR, 'tsne_clusters.png')
                            plt.savefig(tsne_path)
                            plt.close()

                            analysis_results['elbow_curve'] = elbow_path
                            analysis_results['tsne_clusters'] = tsne_path

                            # Show the t-SNE plot directly in Colab
                            img = plt.imread(tsne_path)
                            plt.figure(figsize=(10, 8))
                            plt.imshow(img)
                            plt.axis('off')
                            plt.title('t-SNE Visualization of Patient Clusters')
                            plt.close()
                        except Exception as e:
                            logger.error(f"Error generating t-SNE visualization: {e}")
                            analysis_results['tsne_error'] = str(e)

                    # Analyze cluster characteristics
                    cluster_profiles = {}
                    for i in range(best_k):
                        cluster_data = db_with_clusters[db_with_clusters['Cluster'] == i]
                        profile = {
                            'size': len(cluster_data),
                            'percent': round(len(cluster_data) / len(database) * 100, 1),
                        }

                        # Check for 'Condition Category' column
                        if 'Condition Category' in cluster_data.columns:
                            top_conditions = cluster_data['Condition Category'].value_counts().head(3)
                            if not top_conditions.empty:
                                profile['top_conditions'] = top_conditions.to_dict()

                        # Add other properties if columns exist
                        if 'Anonymized Age' in cluster_data.columns:
                            age_groups = cluster_data['Anonymized Age'].value_counts()
                            if not age_groups.empty:
                                profile['age_groups'] = age_groups.to_dict()

                        if 'Gender' in cluster_data.columns:
                            gender_ratio = cluster_data['Gender'].value_counts()
                            if not gender_ratio.empty:
                                profile['gender_ratio'] = gender_ratio.to_dict()

                        cluster_profiles[f'Cluster_{i}'] = profile

                    analysis_results['cluster_profiles'] = cluster_profiles
            else:
                analysis_results['clustering_error'] = "Missing required columns for clustering analysis"
        else:
            analysis_results['clustering_error'] = "Not enough data for clustering analysis"

    except Exception as e:
        logger.error(f"Error in cohort analysis: {e}")
        analysis_results['error'] = str(e)

    return analysis_results

# Function to properly process uploaded data
def process_uploaded_file(uploaded_file):
    """Process an uploaded CSV file with patient data"""
    try:
        # Read the uploaded file
        if isinstance(uploaded_file, str):  
            df = pd.read_csv(uploaded_file)
        else:  # File-like object provided
            df = pd.read_csv(uploaded_file)

        # Basic validation
        required_columns = ['Name', 'Age', 'Address', 'Gender', 'Blood', 'Medical Condition']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            return f"Error: Missing required columns: {', '.join(missing_columns)}"

        # Process each row
        results = []
        for _, row in df.iterrows():
            try:
                # Extract data
                name = str(row['Name'])
                age = str(row['Age'])
                address = str(row['Address'])
                gender = str(row['Gender'])
                blood = str(row['Blood'])
                condition = str(row['Medical Condition'])
                notes = str(row['Notes']) if 'Notes' in row and not pd.isna(row['Notes']) else ""

                # Anonymize and save
                result = save_anonymized_data(name, age, address, gender, blood, condition, notes)

                if isinstance(result, pd.DataFrame):
                    results.append(f"Successfully processed record for {name}")
                else:
                    results.append(f"Error processing {name}: {result}")

            except Exception as e:
                results.append(f"Error processing row: {str(e)}")

        return {
            "status": "success" if any("Successfully" in r for r in results) else "error",
            "processed": len(results),
            "details": results
        }

    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        return f"Error processing file: {str(e)}"

# Function to create a complete Gradio interface
def create_interface():
    """Create a Gradio interface for the AnonyMed system"""
    with gr.Blocks(title="AnonyMed - Medical Data Anonymization System") as app:
        gr.Markdown("# AnonyMed: Medical Data Anonymization System")
        gr.Markdown("Safely anonymize and analyze medical data while preserving privacy")

        with gr.Tabs():
            # Tab 1: Enter Individual Patient Data
            with gr.TabItem("Enter Patient Data"):
                with gr.Row():
                    with gr.Column():
                        name_input = gr.Textbox(label="Full Name")
                        age_input = gr.Number(label="Age", minimum=0, maximum=120)
                        address_input = gr.Textbox(label="Address", lines=2)

                    with gr.Column():
                        gender_input = gr.Dropdown(label="Gender", choices=["Male", "Female", "Other", "Prefer not to say"])
                        blood_input = gr.Dropdown(label="Blood Type", choices=["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", "Unknown"])
                        condition_input = gr.Textbox(label="Medical Condition")

                notes_input = gr.Textbox(label="Additional Notes (Optional)", lines=3)

                submit_button = gr.Button("Anonymize & Save Data")
                result_output = gr.Dataframe(label="Anonymized Data")

                submit_button.click(
                    fn=save_anonymized_data,
                    inputs=[name_input, age_input, address_input, gender_input,
                            blood_input, condition_input, notes_input],
                    outputs=result_output
                )

            # Tab 2: Privacy Analysis
            with gr.TabItem("Privacy Analysis"):
                with gr.Row():
                    k_value = gr.Slider(label="K-Anonymity Parameter", minimum=1, maximum=10, value=5, step=1)
                    check_button = gr.Button("Check K-Anonymity")

                k_result = gr.JSON(label="K-Anonymity Results")

                check_button.click(
                    fn=check_k_anonymity,
                    inputs=k_value,
                    outputs=k_result
                )

                privacy_button = gr.Button("Generate Privacy Dashboard")
                privacy_output = gr.JSON(label="Privacy Dashboard Results")

                privacy_button.click(
                    fn=generate_privacy_dashboard,
                    inputs=[],
                    outputs=privacy_output
                )

            # Tab 3: Visualizations
            with gr.TabItem("Visualizations"):
                viz_button = gr.Button("Generate Visualizations")
                viz_output = gr.Gallery(label="Data Visualizations")

                viz_button.click(
                    fn=visualize_anonymization,
                    inputs=[],
                    outputs=viz_output
                )

            # Tab 4: Cohort Analysis
            with gr.TabItem("Cohort Analysis"):
                cohort_button = gr.Button("Perform Cohort Analysis")
                cohort_output = gr.JSON(label="Cohort Analysis Results")

                cohort_button.click(
                    fn=perform_cohort_analysis,
                    inputs=[],
                    outputs=cohort_output
                )

            # Tab 5: Download Data - Fixed
            with gr.TabItem("Download Data"):
                gr.Markdown("### Download Anonymized Patient Data")
                gr.Markdown("Click the button below to prepare your data for download.")
                
                export_button = gr.Button("Prepare Download", variant="primary")
                status = gr.Markdown("Ready to prepare your data")
                file_output = gr.File(label="Download File", visible=False)
                
                # Fixed event handler
                export_button.click(
                    fn=export_anonymized_data_for_download,
                    inputs=[],
                    outputs=[file_output, status, file_output]
                )

            # Tab 6: Authorized Access (would require authentication in production)
            with gr.TabItem("Authorized Access"):
                gr.Markdown("⚠️ This feature is for demonstration purposes only and would require authentication in a production system.")

                with gr.Row():
                    record_id = gr.Textbox(label="Record ID")
                    purpose = gr.Textbox(label="Access Purpose")

                access_button = gr.Button("Request Access")
                access_output = gr.Dataframe(label="Original Record Data")

                access_button.click(
                    fn=authorize_access,
                    inputs=[record_id, purpose],
                    outputs=access_output
                )

    return app

# Run the interface if this script is executed directly
if __name__ == "__main__":
    app = create_interface()
    app.launch(share=True)
