# 🏥 Health Risk Prediction

A machine learning-based **Health Risk Prediction System** that predicts a patient's health risk level based on vital health parameters such as respiratory rate, oxygen saturation, blood pressure, heart rate, temperature, consciousness, and oxygen support.

The project uses a **Decision Tree Classifier** for prediction and provides an interactive **Streamlit web application** for users to enter health information and receive a predicted risk level.

> **Note:** This project is intended for educational and demonstration purposes only. It is not a medical diagnostic system and should not be used as a substitute for professional medical advice.

---

## 📌 Project Overview

The Health Risk Prediction system analyzes multiple patient health parameters and classifies the patient's condition into different risk levels such as:

* 🟢 **Low**
* 🟡 **Medium**
* 🔴 **High**

The machine learning model is trained using a healthcare dataset and saved using `joblib`. The trained model is then loaded into a Streamlit application where users can enter patient information and obtain a prediction.

---

## 🎯 Objectives

* Predict health risk levels using machine learning.
* Analyze important patient vital signs.
* Build an easy-to-use web interface using Streamlit.
* Convert categorical health information into numerical values using Label Encoding.
* Train and evaluate a Decision Tree classification model.
* Save and reuse the trained machine learning model.

---

## 🧠 Machine Learning Workflow

The project follows the following workflow:

```text
Health Risk Dataset
        ↓
Data Loading
        ↓
Data Inspection
        ↓
Missing Value Checking
        ↓
Label Encoding
        ↓
Feature Selection
        ↓
Train/Test Split
        ↓
Decision Tree Classifier
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Save Model using Joblib
        ↓
Streamlit Web Application
        ↓
User Input
        ↓
Health Risk Prediction
```

---

## 📊 Features Used

The model uses the following features:

| Feature             | Description                                     |
| ------------------- | ----------------------------------------------- |
| `Respiratory_Rate`  | Number of breaths per minute                    |
| `Oxygen_Saturation` | Percentage of oxygen saturation in blood        |
| `O2_Scale`          | Oxygen requirement/support scale                |
| `Systolic_BP`       | Systolic blood pressure                         |
| `Heart_Rate`        | Heart beats per minute                          |
| `Temperature`       | Body temperature                                |
| `Consciousness`     | Patient consciousness state                     |
| `On_Oxygen`         | Whether the patient is receiving oxygen support |

### Target Variable

```text
Risk_Level
```

The target variable represents the predicted health risk category.

---

## 🤖 Machine Learning Model

### Decision Tree Classifier

The project uses the `DecisionTreeClassifier` from Scikit-learn.

Decision Trees classify data by making a series of decisions based on feature values. They are suitable for this project because they can handle classification problems and are relatively easy to interpret.

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
```

---

## 🔤 Label Encoding

The `Consciousness` column contains categorical values.

A `LabelEncoder` is used to convert these categories into numerical values that can be processed by the machine learning model.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['Consciousness'] = le.fit_transform(
    df['Consciousness']
)
```

The encoder is also saved so that the same encoding can be used when making predictions in the Streamlit application.

---

## 📈 Model Evaluation

The model is evaluated using:

### Accuracy

```python
accuracy_score(y_test, y_pred)
```

### Classification Report

```python
classification_report(y_test, y_pred)
```

The classification report provides metrics such as:

* Precision
* Recall
* F1-score
* Support

---

## 🖥️ Streamlit Application

The project includes an interactive Streamlit interface.

Users can enter:

* Respiratory Rate
* Oxygen Saturation
* O2 Scale
* Systolic Blood Pressure
* Heart Rate
* Temperature
* Consciousness
* Oxygen Support

After clicking:

```text
Predict Health Risk
```

the application sends the entered values to the trained model and displays the predicted risk level.

---

## 🗂️ Project Structure

```text
Health-Risk-Prediction/
│
├── Health_Risk_Dataset.csv
│
├── train_model.py
│
├── app.py
│
├── HealthRiskModel.pkl
│
├── labelEncoder.pkl
│
├── image.png
│
├── requirements.txt
│
└── README.md
```

### File Description

| File                      | Purpose                           |
| ------------------------- | --------------------------------- |
| `Health_Risk_Dataset.csv` | Dataset used for training         |
| `train_model.py`          | ML model training and evaluation  |
| `app.py`                  | Streamlit web application         |
| `HealthRiskModel.pkl`     | Saved Decision Tree model         |
| `labelEncoder.pkl`        | Saved Label Encoder               |
| `image.png`               | Background image for Streamlit UI |
| `requirements.txt`        | Required Python packages          |
| `README.md`               | Project documentation             |

---

## ⚙️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* Decision Tree Classifier
* Label Encoding

### Data Processing

* Pandas

### Model Persistence

* Joblib

### Web Application

* Streamlit

### Other

* Base64
* HTML/CSS

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Health-Risk-Prediction.git
```

Navigate into the project:

```bash
cd Health-Risk-Prediction
```

---

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install pandas scikit-learn streamlit joblib
```

Or, if a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Step 1 — Train the Model

Run:

```bash
python train_model.py
```

This will:

1. Load the dataset.
2. Check the dataset structure.
3. Check missing values.
4. Encode the `Consciousness` column.
5. Select the required features.
6. Split the data into training and testing sets.
7. Train the Decision Tree model.
8. Evaluate the model.
9. Save the trained model.

The following files will be generated:

```text
HealthRiskModel.pkl
labelEncoder.pkl
```

---

### Step 2 — Run the Streamlit Application

Run:

```bash
streamlit run app.py
```

Streamlit will start the application locally.

Open the URL displayed in the terminal, usually:

```text
http://localhost:8501
```

---

## 🧪 Example Input

Example patient information:

```text
Respiratory Rate     : 24
Oxygen Saturation    : 92
O2 Scale             : 2
Systolic BP          : 105
Heart Rate           : 98
Temperature          : 37.2
Consciousness        : Encoded value
On Oxygen            : 1
```

The trained model will process these values and return a risk category.

Example:

```text
Predicted Risk Level: Medium
```

The actual prediction depends on the trained dataset and model.

---

## 🔮 Prediction Process

The Streamlit application creates the input in the same feature order used during training:

```python
data = [[
    Respiratory_Rate,
    Oxygen_Saturation,
    O2_Scale,
    Systolic_BP,
    Heart_Rate,
    Temperature,
    Consciousness_encode,
    On_Oxygen
]]
```

The model then predicts the risk level:

```python
prediction = model.predict(data)
```

The result is displayed using Streamlit:

```python
st.error()
```

for high risk,

```python
st.warning()
```

for medium risk, and

```python
st.success()
```

for low risk.

---

## 💾 Model Saving

The trained model is saved using Joblib:

```python
joblib.dump(model, 'HealthRiskModel.pkl')
```

The Label Encoder is also saved:

```python
joblib.dump(le, 'labelEncoder.pkl')
```

These files allow the Streamlit application to use the already-trained model without training it every time.

---

## 🔐 Important Considerations

For a production healthcare application, additional considerations would be required, including:

* Proper medical validation
* Larger and clinically representative datasets
* Feature scaling/normalization where appropriate
* Model comparison
* Cross-validation
* Hyperparameter tuning
* Bias and fairness evaluation
* Patient data privacy
* Secure data storage
* Clinical expert validation
* Explainable AI
* Appropriate medical and regulatory review

---

## 🚀 Future Improvements

The project can be extended with:

* 📊 Interactive health dashboards
* 📈 Visualization of patient vital signs
* 🧠 Random Forest and XGBoost model comparison
* ⚙️ Hyperparameter tuning
* 📋 Patient history
* 🔐 User authentication
* 💾 Database integration
* 📱 Responsive UI
* 📄 Downloadable prediction reports
* 📉 Risk probability visualization
* 🤖 Explainable AI using SHAP
* ☁️ Cloud deployment
* 🏥 Integration with healthcare systems

---

## 📌 Disclaimer

This project is developed for **educational and machine learning demonstration purposes**.

The predictions generated by this application should **not be considered medical advice, diagnosis, or treatment recommendations**. Always consult a qualified healthcare professional for medical decisions.

---

## 👩‍💻 Author

**Keertana**

Computer Science and Engineering Student

---

## ⭐ Acknowledgement

This project demonstrates the application of machine learning classification techniques to healthcare-related data and provides an interactive interface for experimenting with health risk prediction.

If you found this project useful, consider giving the repository a ⭐ on GitHub.
