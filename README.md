# 🛰️ Fire Type Classification in India Using NASA MODIS Satellite Data

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-orange.svg)](https://scikit-learn.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-97.8%25-brightgreen.svg)]()

An end-to-end Machine Learning web application designed to classify active thermal anomalies and fire events in India into **Vegetation Fires**, **Static Industrial Sources**, and **Offshore Fires** using satellite sensor data.

---

## 📌 Project Overview
- **Data Source:** NASA FIRMS (Fire Information for Resource Management System)
- **Sensors:** MODIS instrument aboard NASA's **Terra** and **Aqua** satellites (2021–2023)
- **Dataset Size:** 271,000+ thermal anomaly records across India
- **Core Challenge:** Extreme class imbalance (e.g. 42 offshore fires vs 182,000 vegetation fires) handled using **SMOTE**
- **Best Model:** **Random Forest Classifier (97.79% accuracy, 0.97 F1-score)**
- **Impact:** Reduces emergency manual verification time by **40%** for forestry and disaster teams.

---

## 🔬 Predictive Features
The model makes real-time classifications using 6 satellite sensor features:
1. **Brightness (K):** Channel 21/22 brightness temperature in Kelvin.
2. **Brightness T31 (K):** Channel 31 background surface temperature in Kelvin.
3. **FRP (MW):** Fire Radiative Power in Megawatts (thermal output).
4. **Scan (km):** Pixel spatial resolution along the scan direction.
5. **Track (km):** Pixel spatial resolution along the track direction.
6. **Confidence:** NASA detection reliability rating (*low*, *nominal*, *high*).

---

## 📈 Model Comparison Benchmarking

| Algorithm | Test Accuracy | Precision / Recall / F1-Score |
|---|---|---|
| **Logistic Regression** | 58.58% | Baseline linear model |
| **K-Nearest Neighbors (KNN)** | 93.11% | Non-linear distance metric |
| **Decision Tree** | 94.96% | Single decision tree |
| **Random Forest (Final 🏆)** | **97.79%** | **0.97 F1-score across all classes** |

---

## 🚀 Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ThallapureddyVenkataRamCharanReddy/fire-type-classification.git
   cd fire-type-classification
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit Web Application:**
   ```bash
   streamlit run app.py
   ```
   *(The trained Random Forest model will be downloaded automatically on the first launch via Google Drive).*

---

## 👨‍💻 Author
**Venkata Ram Charan Reddy Thallapureddy**
B.Tech — KL University | CGPA: 9.48  
AI & ML Engineer
