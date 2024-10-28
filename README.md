# **CS506 Midterm Project - Movie Review Score Prediction**

### **Author**: Adi Bhan  
### **Course**: CS506 - Fall 2024  

---

## **Overview**
This project predicts movie review scores using a **RandomForestClassifier** model trained on a dataset of review text and metadata. Initially, **K-Nearest Neighbors** was tested, but **RandomForestClassifier** was selected for improved runtime and built-in feature importance evaluation, which enhanced predictive accuracy. The model uses a combination of sentiment scores, adjective counts, text complexity, and temporal data to better capture review sentiments.

---

## **Project Structure**
- **model.py**: Core model code for data processing, feature engineering, and model training.
- **graphs/**: Directory containing feature correlation and importance graphs.
- **data/**: Folder with various CSV files for training and testing data.
- **report.pdf**: Full report detailing feature selection, accuracy, and model performance.
