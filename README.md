# Credit_Card_Fraud_Detection
This project aims to build a machine learning model to detect fraudulent credit card transactions based on a dataset containing transaction details.

## 📊 Dataset
The dataset used for this project is a highly imbalanced dataset containing credit card transactions, with the following characteristics:

Features: Time, V1 to V28 (PCA-transformed), and Amount
Target: Class (0 = Non-Fraudulent, 1 = Fraudulent)
## 🛠️ Data Preprocessing
Handling Missing Values:

Checked for null values; none were found in the dataset.
Feature Scaling:

The Amount column was scaled using StandardScaler to bring it to a standard range.
PCA-transformed features (V1 to V28) were already standardized.
Train-Test Split:

Split the data into 80% training and 20% testing using train_test_split().
Addressing Class Imbalance:

The dataset is highly imbalanced, so SMOTE (Synthetic Minority Over-sampling Technique) was applied to balance the classes.
## 🤖 Model Training
A Logistic Regression Model was trained using the preprocessed data.

### Steps:

Imported the Model:
python
Copy code
from sklearn.linear_model import LogisticRegression
Model Initialization:
python
Copy code
model = LogisticRegression()
Model Training:
python
Copy code
model.fit(X_train, y_train)
## 🧪 Model Evaluation
Evaluation metrics were used to assess the model's performance:

Accuracy Score
Confusion Matrix
Classification Report (Precision, Recall, F1-Score)

## 🗂️ Saving the Model
The trained model was saved using Pickle for later use.

## 📝 Key Libraries Used
pandas
numpy
scikit-learn
imbalanced-learn (for SMOTE)

This summary captures the end-to-end process you followed in your Credit Card Fraud Detection Project.
