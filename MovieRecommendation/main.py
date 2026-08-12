import streamlit as st
import joblib

model = joblib.load('movie_model.pkl')
X = joblib.load('movie features.pkl')
movie_data = joblib.load('movie_data.pkl')

st.title("🎬Movie Recommendation")

movie = st.selectbox(
  'choose a movie:',
  movie_data['title'].tolist()
)

if st.button('Movie Recommendation'):
  movie_index  = movie_data[
    movie_data['title'] == movie
  ].index[0]

  distnaces,indices = model.kneighbors(
    X.iloc[[movie_index]]
  )

  recommendations = movie_data.iloc[indices[0]]
  st.subheader('Recommended movies')

  for title in recommendations['title']:
    st.write(title)
