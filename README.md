# AnonyMed-Medical-Data-Anonymization-System

AnonyMed is a comprehensive tool for anonymizing, managing, and analyzing medical data while preserving patient privacy. This system implements various privacy-preserving techniques including **k-anonymity** and **pseudonymization** to help healthcare organizations handle sensitive patient information securely.

---

## 📌 Overview

The medical field requires special attention to privacy due to the sensitive nature of health data. AnonyMed provides tools to:

- Anonymize individual patient records  
- Process uploaded CSV files containing patient information  
- Perform privacy analysis and risk scoring  
- Visualize anonymized data patterns  
- Conduct cohort analysis on anonymized datasets  
- Export privacy-preserving datasets for research  

---

## 🔑 Core Features

### 🛡️ Data Anonymization
- Consistent pseudonymization of patient identifiers  
- Age bracketing to prevent exact age identification  
- Location generalization for addresses  
- Medical condition categorization  
- Automated PHI (Protected Health Information) detection in free text  
- Secure encryption of original data with access controls  

### 🔍 Privacy Analysis
- K-anonymity verification and reporting  
- Re-identification risk scoring  
- Privacy dashboard with visualizations  
- Data utility preservation metrics  

### 📊 Data Visualization & Analysis
- Distribution visualizations for demographic data  
- Cohort analysis capabilities  
- Patient clustering with t-SNE visualization  
- Privacy risk distribution charts  

---

## 🛠️ Implementation Notes

### 📦 Dependencies

The system requires several Python libraries:

- `pandas`, `numpy` — Data manipulation  
- `matplotlib`, `seaborn` — Visualization  
- `scikit-learn` — Clustering and dimensionality reduction  
- `gradio` — User interface  
- `cryptography` — Secure data storage  
- `spaCy` *(optional)* — PHI detection using NLP  

### 💾 Data Storage

- Anonymized data is stored in CSV format  
- Original identifiers are encrypted and stored separately  
- Encryption keys are securely managed  

### 🧩 Privacy Mechanisms

1. **Pseudonymization**: Replaces identifiers with consistent pseudonyms  
2. **Generalization**: Reduces precision of values (e.g., age ranges, location regions)  
3. **Categorization**: Groups medical conditions into broader categories  
4. **K-anonymity**: Ensures each record shares attributes with at least *k-1* other records  
5. **Access Control**: Encryption and purpose logging for authorized data recovery  

---

## ⚠️ Limitations

1. **Basic NLP capabilities** — spaCy may miss uncommon/contextual PHI  
2. **Limited differential privacy** — Formal DP techniques not implemented  
3. **Simplistic risk scoring** — Only k-anonymity based  
4. **Limited attribute inference protection**  
5. **Simplified cohort analysis** — Not a full statistical framework  

---

## 🔐 Security Considerations

- Additional security measures should be implemented in production  
- Authorized Access should include proper authentication & authorization  
- Key management should use secure, dedicated services  
- Logging and audit trails should be comprehensive  
- Regular security audits are strongly recommended  
