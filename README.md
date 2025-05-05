📉 Telecom Customer Churn Prediction

## 📝 Overview
This project analyzes telecom customer data and aims to **predict churn** using data analysis and machine learning techniques. The goal is to identify customers who are likely to leave the service, which can help telecom providers take preventive actions.

---

## 📊 Dataset Summary

The dataset contains the following key information:

- **Demographic Info**: `gender`, `age`, `state`, `city`
- **Telecom Details**: `telecom_partner`, `date_of_registration`, `num_dependents`
- **Usage Behavior**: `calls_made`, `sms_sent`, `data_used`
- **Financial Info**: `estimated_salary`
- **Label**: `churn` (1 = churned, 0 = retained)

Some preprocessing steps:
- Converted `date_of_registration` to datetime format.
- Removed unnecessary features like `customer_id`, `pincode`, and `date_of_registration`.

---

## 📌 Project Goals

- Perform Exploratory Data Analysis (EDA)
- Visualize correlations using heatmaps
- Build a basic machine learning model to predict churn
- Evaluate model performance
- Discuss next steps for improvement

---

## 📈 Model Performance

A baseline classification model was trained to predict churn:

**Classification Report:**
```
           Precision    Recall  F1-Score   Support
       0       0.80      1.00      0.89     38928
       1       0.67      0.00      0.00      9783

Accuracy:           0.80  
Macro Avg F1:       0.44  
ROC-AUC Score:      0.5001
```

> ⚠️ **Insight**: The model struggles to detect churners (class 1), mostly due to class imbalance. Accuracy is misleadingly high due to majority class dominance.

---

### 🔍 Confusion Matrix

To visualize how well the model performs in distinguishing churned vs non-churned customers, we use a confusion matrix:

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

**Interpretation:**
- **True Negatives (Top-left):** Customers correctly predicted as **not churned**
- **True Positives (Bottom-right):** Customers correctly predicted as **churned**
- **False Positives (Top-right):** Non-churners incorrectly predicted as **churned**
- **False Negatives (Bottom-left):** Churners incorrectly predicted as **not churned**

📌 The matrix confirms that the model rarely predicts class `1`, missing nearly all churners.

---

## ⚙️ Tech Stack

- Python
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn

---

## 🚀 Future Improvements

- Apply techniques to handle class imbalance (e.g., SMOTE, class weighting)
- Use more advanced models like Random Forest or XGBoost
- Hyperparameter tuning
- Build an interactive dashboard using Streamlit or Flask

---

## 📁 Structure

```
├── data/
│   └── telecom_churn.csv
├── notebooks/
│   └── telecom_churn_analysis.ipynb
├── outputs/
│   └── plots and results
├── README.md
```

---

## 📬 Contact

For feedback or collaboration: danialmorkos2021@gmail.com
