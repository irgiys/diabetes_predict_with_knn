import pickle 
import streamlit as st
import pandas as pd

model = pickle.load(open('diabetes_model.sav', 'rb'))

st.set_page_config(
    page_title="Prediksi Diabetes | by Irgiys",
    page_icon="🍭",
)

st.markdown(
    """
        # 👨‍⚕️ Prediksi Diabetes
    """
)

with st.sidebar:
    genre = st.radio(
    "Isi inputan prediksi",
    ["Tidak terkena diabetes", "Diabetes"],
    captions = ["Inputan akan terisi dengan contoh data tidak terkena diabetes", "Inputan akan terisi dengan contoh data terkena diabetes"])

if genre == "Diabetes" :
    age = 30
    insulin = 20
    bmi = 51.1
else : 
    age = 22
    insulin = 50
    bmi = 28.1

tab1, tab2 = st.tabs(["Prediksi", "Tentang Web App"])
with tab1:
    st.write("Untuk melakukan prediksi website ini membutuhkan **8 inputan** dengan ketentuan tertentu sehingga menghasilkan prediksi yang lebih akurat.")
    col1, col2 = st.columns(2)
    with col1:
        Age = st.number_input("Umur", 21,81,age)
    with col2: 
        Pregnancies = st.number_input("Jumlah kehamilan yang pernah dialami", 0,17,1)
    
    with col1:
        Glucose = st.number_input("Kadar glukosa dalam darah, diukur dalam mg/dL", 0,199,85)
    with col2: 
        BloodPressure = st.number_input("Tekanan darah (mm Hg)", 0,122,66)

    with col1:
        SkinThickness = st.number_input("Ketebalan lipatan kulit trisep (mm)", 0,99,29)
    with col2: 
        Insulin = st.number_input("Insulin serum (mu U/ml) ", 0,846,insulin)

    with col1:
        BMI = st.number_input("Body Mass Index (Berat dalam kg/(tinggi dalam  m)²)", 0.0,67.1,bmi)
    with col2: 
        DiabetesPedigreeFunction = st.number_input("Fungsi pedigri diabetes", 0.078,2.42,0.351)

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

with tab2:
    st.write("Web App ini diciptakan untuk menjadi alat yang membantu individu dalam memeriksa kemungkinan adanya diabetes secara cepat dan mudah. Pentingnya untuk diingat bahwa Web App ini bukan pengganti dari konsultasi medis yang sesungguhnya.")
    
    df = pd.read_csv("diabetes.csv")
    st.markdown("#### Sampel 5 data dalam dataset")
    st.write(df.head(5))