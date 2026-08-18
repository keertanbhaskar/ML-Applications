import streamlit as st

if "counter" not in st.session_state:
  st.session_state.counter = 0
if st.button('up'):
  st.write(st.session_state.counter)
  st.session_state.counter += 1
if st.button('reset'):
  st.session_state.counter = 0