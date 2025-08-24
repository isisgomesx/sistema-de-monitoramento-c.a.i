import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

#python -m streamlit run app.py


modelo = joblib.load("modelo_xgboost.joblib")


st.set_page_config(page_title="Sistema de Monitoramento - C.A.I", layout="centered")


st.markdown("""
    <style>
    .stButton>button {
        background-color: #7843e6;
        color: white;
        height: 40px;
        border-radius: 10px;
        border: none;
        font-size: 16px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #5a2ebc;
        color: white;
    }
    .stTextInput>div>div>input {
        height: 40px;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Sistema de Monitoramento - C.A.I")
st.markdown("Preencha os dados abaixo para simular o risco de saúde.")

valores_padrao = {
    "Heart Rate": 70,
    "Body Temperature": 36.5,
    "Oxygen Saturation": 98,
    "Systolic Blood Pressure": 120,
    "Diastolic Blood Pressure": 80,
    "Respiratory Rate": 16
}

with st.form("form_simulacao"):
    col1, col2 = st.columns(2)
    with col1:
        idade = st.number_input(
            "Idade (anos)", 
            min_value=0, 
            max_value=120, 
            value=30
            )
        peso = st.number_input(
            "Peso (kg)", 
            min_value=0.0, 
            max_value=300.0, 
            value=70.0
            )
        altura = st.number_input(
            "Altura (m)", 
            min_value=0.5, 
            max_value=2.5, 
            value=1.70, 
            format="%.2f"
            )
        fc = st.number_input(
            "Frequência Cardíaca (bpm)", 
            min_value=0, 
            max_value=250, 
            value=70,
            help="Número de batimentos por minuto. Normal em repouso: 60-100 bpm."
            )
        pressao_s = st.number_input(
            "Pressão Sistólica (mmHg)", 
            min_value=0, 
            max_value=250, 
            value=120,
            help="Pressão máxima nas artérias quando o coração bate. Normal: 110-120    mmHg."
            )
    with col2:
        pressao_d = st.number_input(
            "Pressão Diastólica (mmHg)", 
            min_value=0, 
            max_value=200, 
            value=80,
            help="Pressão mínima nas artérias entre os batimentos. Normal: 70-80 mmHg."
            )
        ox = st.number_input(
            "Oxigenação (%)", 
            min_value=0, 
            max_value=100, 
            value=98,
            help="Nível de oxigênio no sangue. Normal: 95-100%."
            )
        temp = st.number_input(
            "Temperatura Corporal (°C)", 
            min_value=30.0, 
            max_value=45.0, 
            value=36.5,
            format="%.1f",
            help="Temperatura do corpo. Normal: 36.1-37.2 °C."
            )
        rp = st.number_input(
            "Frequência Respiratória", 
            min_value=0, 
            max_value=50, 
            value=16,
            help="Número de respirações por minuto. Normal: 12-20 rpm.")
        imc = peso / (altura ** 2)

    submitted = st.form_submit_button("Simular 💜")

if submitted:
    dados = pd.DataFrame([{
        "Heart Rate": fc,
        "Body Temperature": temp,
        "Oxygen Saturation": ox,
        "Systolic Blood Pressure": pressao_s,
        "Diastolic Blood Pressure": pressao_d,
        "Age": idade,
        "Weight (kg)": peso,
        "Height (m)": altura,
        "Derived_BMI": imc,
        "Respiratory Rate": rp
    }])

    score = modelo.predict(dados)[0]
    prob = modelo.predict_proba(dados)[0][1]

    if score == 1:
        risco_texto = "Risco alto — procure ajuda médica imediatamente!"
        cor = "#FF4B4B"
    else:
        risco_texto = "Risco baixo — situação estável."
        cor = "#4BB543"

    st.markdown(f"<h2 style='color:{cor}'>Score de Risco: {score} ({prob*100:.2f}%)</h2>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color:{cor}'>{risco_texto}</h3>", unsafe_allow_html=True)

    categorias = ["Freq. Cardíaca", "Pressão Sistólica", "Pressão Diastólica", "Oxigenação", "Temperatura", "Respiratório"]
    valores_simulados = [fc, pressao_s, pressao_d, ox, temp, rp]
    valores_referencia = [
            valores_padrao["Heart Rate"],
            valores_padrao["Systolic Blood Pressure"],
            valores_padrao["Diastolic Blood Pressure"],
            valores_padrao["Oxygen Saturation"],
            valores_padrao["Body Temperature"],
            valores_padrao["Respiratory Rate"]
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(categorias))
    largura = 0.35

    ax.bar([p - largura/2 for p in x], valores_referencia, width=largura, color='gray', alpha=0.6, label="Padrão")
    ax.bar([p + largura/2 for p in x], valores_simulados, width=largura, color=cor, alpha=0.8, label="Simulado")
    ax.set_xticks(x)
    ax.set_xticklabels(categorias, rotation=25)
    ax.set_ylabel("Valores")
    ax.set_title("Comparação: Valores Simulados x Padrão estável")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    st.pyplot(fig)