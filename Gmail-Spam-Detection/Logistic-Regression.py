from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score,precision_score,recall_score
import joblib

df = pd.read_csv("spam.csv",encoding='latin-1') 
#error occurred =>UnicodeDecodeError: 'utf-8' codec can't decode bytes

print(df.head(4))
print(df.isnull().sum())

df = df.drop(df.columns[2:5],axis=1)
print(df.isnull().sum())

messages = df['v2']
y = df['v1']

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(messages)


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

prediction = model.predict(vectorizer.transform(['Congratulations! You have won a free prize. Click now to claim your reward.']))
print('Is spam:',prediction[0])

# accuracy score 
# print("acc score:",accuracy_score(y_test,y_pred))


joblib.dump(model,"LogisticReg.pkl")
joblib.dump(vectorizer,"LogVectorizer.pkl")

