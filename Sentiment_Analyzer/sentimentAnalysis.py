# sentiment analyzer

import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv('train.csv',encoding='latin-1')

print(data.shape)
print(data.info())
print(data.isnull().sum())

le = LabelEncoder()
data['sentiment'] = le.fit_transform(data['sentiment'])
data['selected_text'] = data['selected_text'].fillna('')
feature = data['selected_text']
target = data['sentiment']

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(feature)
y = target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = MultinomialNB()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

msg = ['hey it was good day']
X_msg = vectorizer.transform(msg)
prediction = model.predict(X_msg)
print("the sentiment:",le.inverse_transform(prediction)[0])


# accuracy score
print("accuracy is:",accuracy_score(y_test,y_pred))

joblib.dump(model,'sentimentAnalyzeModel.pkl')
joblib.dump(vectorizer,'Vecorizer.pkl')
joblib.dump(le,'labelEncoder.pkl')