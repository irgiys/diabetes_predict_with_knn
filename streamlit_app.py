import pickle 
import streamlit as st


model = pickle.load(open('diabetes_model.sav', 'rb'))

st.set_page_config(
    page_title="Prediksi Diabetes",
    page_icon="🍭",
)

st.markdown(
    """
        # 👨‍⚕️ Prediksi Diabetes
        Untuk melakukan prediksi website ini membutuhkan **8 inputan** dengan ketentuan tertentu sehingga menghasilkan prediksi yang lebih akurat.
    """
)



col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Umur", 21,81,30)
with col2: 
    Pregnancies = st.number_input("Jumlah kehamilan yang pernah dialami", 0,17,1)
  
with col1:
    Glucose = st.number_input("Kadar glukosa dalam darah, diukur dalam mg/dL", 0,199,90)
with col2: 
    BloodPressure = st.number_input("Tekanan darah (mm Hg)", 0,122,61)

with col1:
    SkinThickness = st.number_input("Ketebalan lipatan kulit trisep (mm)", 0,99,0)
with col2: 
    Insulin = st.number_input("Insulin serum (mu U/ml) ", 0,846,79)

with col1:
    BMI = st.number_input("Body Mass Index (Berat dalam kg/(tinggi dalam  m)²)", 0,67,33)
with col2: 
    DiabetesPedigreeFunction = st.number_input("Fungsi pedigri diabetes", 94,200,150)

if st.button("Prediksi penyakit Diabetes"):
    heart_disease_predict = model.predict([[
                                            Pregnancies,
                                            Glucose,
                                            BloodPressure,
                                            SkinThickness,
                                            Insulin,
                                            BMI,
                                            DiabetesPedigreeFunction,
                                            Age
                                            ]])
    if(heart_disease_predict[0]==0):
        st.success("Pasien tidak terkena diabetes",icon="☺️")
    else :
        st.warning("Pasien terkena diabetes", icon="😔")
        