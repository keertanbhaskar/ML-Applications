# Health risk prediction
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import joblib

df = pd.read_csv('Health_Risk_Dataset.csv')
print(df.shape)
print(df.columns)

print(df.isnull().sum())

print(df.info())

le = LabelEncoder()
df['Consciousness'] = le.fit_transform(df['Consciousness'])

X = df [['Respiratory_Rate','Oxygen_Saturation','O2_Scale',
          'Systolic_BP','Heart_Rate','Temperature','Consciousness','On_Oxygen']]

y = df['Risk_Level']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print('accuracy:',accuracy_score(y_test,y_pred))
print("classification report:")
print(classification_report(y_test,y_pred))


# new data for prediction
test_data = [[24,92,2,105,98,37.2,0,1]]
prediction = model.predict(test_data)
print("Predicted Risk Level:", prediction[0])

joblib.dump(model,'HealthRiskModel.pkl')
joblib.dump(le,'labelEncoder.pkl')

