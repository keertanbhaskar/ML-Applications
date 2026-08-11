import streamlit as st
import joblib

logistic_model = joblib.load("LogisticReg.pkl")
naive_bayes_model = joblib.load("NaiveBayes.pkl")
vectorizer = joblib.load("LogVectorizer.pkl")

st.title("Gmail Spam Detection")


st.sidebar.title("Model Selection")

model_choice = st.sidebar.radio(
    "Choose Model:",
    ["Logistic", "NaiveBayes"]
)


message = st.text_input(
    "Enter the message:",
    placeholder="Enter message...")



if st.button("Check Message"):
        msg_vectorizer = vectorizer.transform([message])
        if model_choice == "Logistic":
            prediction = logistic_model.predict(msg_vectorizer)[0]

        else:
            prediction = naive_bayes_model.predict(msg_vectorizer)[0]
        st.write("Model Used:",model_choice)
        if prediction == "spam":
            st.error("Spam message")

        else:
            st.success("Not a spam message")