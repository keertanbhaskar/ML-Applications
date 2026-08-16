import streamlit as st
import joblib

model = joblib.load('sentimentAnalyzeModel.pkl')
vectorizer = joblib.load('Vecorizer.pkl')
le = joblib.load('labelEncoder.pkl')

st.title("Sentiment Analyzer")

text = st.text_input("Enter the text...")

if st.button("Click Me"):

    if text.strip() == "":
        st.warning("Please enter some text.")

    else:
        X = vectorizer.transform([text])

        prediction = model.predict(X)

        sentiment = le.inverse_transform(prediction)[0]

        st.success(f"Sentiment: {sentiment}")