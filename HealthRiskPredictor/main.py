import base64
import streamlit as st
import joblib

# background image
def set_background(image_file):

    with open(image_file, "rb") as file:
        encoded_image = base64.b64encode(file.read()).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """

    st.markdown(css, unsafe_allow_html=True)


set_background("image.png")

model = joblib.load('HealthRiskModel.pkl')
le = joblib.load('labelEncoder.pkl')

st.title('Health Risk Prediction')

# X = df [['Respiratory_Rate','Oxygen_Saturation','O2_Scale',
        #  'Systolic_BP','Heart_Rate','Temperature','Consciousness','On_Oxygen']]

Respiratory_Rate = st.number_input("Respiratory_Rate",min_value=0,max_value=50)
Oxygen_Saturation = st.number_input("Oxygen Saturation",min_value=0,max_value=100)
O2_Scale = st.number_input("O2 Scale",min_value=0,max_value=10)
Systolic_BP = st.number_input("Systolic BP",min_value=0,max_value=250,)
Heart_Rate = st.number_input("Heart Rate",min_value=0,max_value=250)
Temperature = st.number_input("Temperature",min_value=30.0,max_value=45.0,step=0.1)

Consciousness = st.selectbox(
  'Consciousness',
  le.classes_
)
Consciousness_encode = le.transform([Consciousness])[0]
On_Oxygen = st.selectbox(
    "On Oxygen",
    [0, 1]
)

if st.button('Predict Health Risk'):
  data = [[Respiratory_Rate,Oxygen_Saturation,O2_Scale,Systolic_BP,Heart_Rate,Temperature,Consciousness_encode,On_Oxygen]]
  prediction = model.predict(data)
  if prediction == 'High':
      st.error(prediction[0])
  elif prediction == 'Medium':
     st.warning(prediction[0])
  else:
     st.success(prediction[0])


