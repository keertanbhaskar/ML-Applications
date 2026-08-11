from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("spam.csv",encoding='latin-1')
df = df.drop(df.columns[2:5],axis=1)
print(df.isnull().sum())

vectorizer = CountVectorizer()

messages = df['v2']
y = df['v1']

X = vectorizer.fit_transform(messages)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = MultinomialNB()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

msg = 'Congratulations! You have won a free prize. Click now to claim your reward.'
X_msg = vectorizer.transform([msg])

print("Is Spam:",model.predict(X_msg)[0])
print("Acc Score:",accuracy_score(y_test,y_pred))

joblib.dump(model,'NaiveBayes.pkl')
joblib.dump(vectorizer,"NaiveBayesVectorizer.pkl")
