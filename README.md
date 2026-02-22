# 🏦 HDFC Loan Approval AI: Ensemble Learning Edition

### 🔗 **Live Demo:** [Check Loan Eligibility (98.7% Accuracy)](https://hdfc-loan-predictor-djb6b6yumfzbfdrdittpeg.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-v1.20+-red.svg)
![Random Forest](https://img.shields.io/badge/Model-Random%20Forest-green.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-98.7%25-brightgreen.svg)

---

## 📌 Project Overview
This project is an advanced **Loan Sanctioning System** designed for HDFC Bank. After rigorous comparative analysis between multiple algorithms, we implemented a **Random Forest Classifier** that achieves a state-of-the-art accuracy of **98.7%**. The system automates credit risk assessment by analyzing financial stability, asset collateral, and credit history.

## 🚀 Model Performance Comparison
We evaluated three different modeling approaches to find the best balance between precision and reliability:

| Algorithm | Accuracy | Technical Highlights |
| :--- | :--- | :--- |
| **Random Forest** | **98.7%** | **Ensemble Learning, Bagging, Entropy-based Splits** |
| Gaussian Naive Bayes | 93.0% | Probabilistic Approach based on Bayes' Theorem |
| Logistic Regression | 90.2% | Linear Binary Classification with Sigmoid Function |

---

## 🧠 Technical Deep Dive

### 1. The Champion: Random Forest
* **Bagging (Bootstrap Aggregating):** Reduces variance by training multiple decision trees on different data subsets.
* **Entropy & Information Gain:** Uses mathematical impurity measures to find the best features for splitting data.
* **Why it won?** Handles non-linear relationships and outliers much better than linear models.

### 2. Feature Engineering: The Asset Logic
Instead of processing assets individually, we engineered a **Total Wealth Metric**:
$$Total\ Assets = Residential + Commercial + Luxury\ Assets$$
This provides a stronger collateral signal to the model, significantly boosting prediction stability.

---

## 🛠 Tech Stack
- **Frontend:** Streamlit (Custom HDFC Blue & Red Theme)
- **Machine Learning:** Scikit-Learn
- **Core Models:** Random Forest, GaussianNB, Logistic Regression
- **Data Scaling:** StandardScaler (Z-score Normalization)
- **Serialization:** Joblib

---

## 👥 The Visionary Team

| Role | Name |
| :--- | :--- |
| **🚀 Team Lead** | **Akshat Gupta** |
| Data Scientist | Mohak Sharma |
| ML Engineer | Jasmine Kaur |
| Risk Analyst | Manav Kaushik |

---

## ⚙️ How to Run Locally
1. **Clone the Repo:**
   ```bash
   git clone [https://github.com/your-username/HDFC-Loan-Predictor.git](https://github.com/your-username/HDFC-Loan-Predictor.git)
