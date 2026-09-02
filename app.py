import os
import numpy as np
import joblib
from PIL import Image
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

IMG_SIZE = 32


def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "melanoma_logistic_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    return model, scaler, label_encoder


def preprocess_image(uploaded_file, model):
    try:
        uploaded_file.seek(0)
        with Image.open(uploaded_file) as img:
            img.verify()
        uploaded_file.seek(0)
        img = Image.open(uploaded_file).convert("RGB")
    except Exception:
        raise ValueError("El archivo no es una imagen válida. Asegúrate de subir un archivo en formato JPG, JPEG o PNG.")
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized).flatten().reshape(1, -1)
    X_scaled = model["scaler"].transform(img_array)
    return img, img_resized, X_scaled


def inject_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

        .stApp {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            background-attachment: fixed;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .main-title {
            text-align: center;
            background: linear-gradient(90deg, #00d2ff, #3a7bd5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            text-align: center;
            color: #b8b8d1;
            font-size: 1.1rem;
            font-weight: 300;
            margin-bottom: 2rem;
            letter-spacing: 1px;
        }

        .upload-area {
            background: rgba(255,255,255,0.06);
            border: 2px dashed rgba(255,255,255,0.25);
            border-radius: 16px;
            padding: 2.5rem;
            text-align: center;
            backdrop-filter: blur(8px);
        }

        .stFileUploader {
            border-radius: 12px;
        }

        .stFileUploader > div {
            background: transparent !important;
        }

        .stFileUploader section {
            background: rgba(255,255,255,0.05);
            border: 2px dashed rgba(255,255,255,0.3);
            border-radius: 12px;
        }

        .pred-card {
            border-radius: 16px;
            padding: 2rem 2.5rem;
            color: white;
            text-align: center;
            box-shadow: 0 15px 40px rgba(0,0,0,0.35);
            animation: slideIn 0.6s ease;
        }

        @keyframes slideIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .pred-card.benigno {
            background: linear-gradient(135deg, #11998e, #38ef7d);
        }

        .pred-card.maligno {
            background: linear-gradient(135deg, #cb2d3e, #ef473a);
        }

        .pred-label {
            font-size: 2.2rem;
            font-weight: 700;
        }

        .pred-desc {
            font-size: 1rem;
            margin-top: 0.5rem;
            opacity: 0.95;
        }

        .conf-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
            margin-top: 1.5rem;
        }

        .conf-cell {
            background: rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
        }

        .conf-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: white;
        }

        .conf-label {
            font-size: 0.75rem;
            color: #d0d0e8;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .sidebar-title {
            color: #00d2ff;
            font-weight: 600;
        }

        .footer {
            text-align: center;
            color: #77779e;
            font-size: 0.8rem;
            margin-top: 2rem;
            border-top: 1px solid rgba(255,255,255,0.1);
            padding-top: 1.5rem;
        }

        .metric-box {
            background: rgba(255,255,255,0.06);
            border-radius: 10px;
            padding: 0.8rem;
            text-align: center;
        }

        .metric-title {
            font-size: 0.7rem;
            color: #a0a0c5;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .metric-val {
            font-size: 1.6rem;
            font-weight: 700;
            color: white;
        }

        .dropdown-container {
            background: rgba(255,255,255,0.05) !important;
        }

        /* Hide default Streamlit footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {background: transparent !important;}
    </style>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="Melanoma Detector AI",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    inject_css()

    st.markdown('<div class="main-title">🩺 Melanoma Detector AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Clasificación de Lesiones Dérmicas · Benigno vs Maligno</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div class="sidebar-title">📊 Acerca del Modelo</div>', unsafe_allow_html=True)
        st.markdown("""
        Este sistema utiliza **Regresión Logística** para clasificar lesiones de piel como:
        """)
        st.markdown("""
        - 🟢 **Benigno (0)**: Lesión no cancerosa
        - 🔴 **Maligno (1)**: Lesión cancerosa (melanoma)
        """)
        st.divider()
        st.markdown("""
        **Métricas del modelo (test):**
        """)
        col1, col2 = st.columns(2)
        col1.markdown('<div class="metric-box"><div class="metric-title">Accuracy</div><div class="metric-val">84.3%</div></div>', unsafe_allow_html=True)
        col2.markdown('<div class="metric-box"><div class="metric-title">AUC-ROC</div><div class="metric-val">92.0%</div></div>', unsafe_allow_html=True)
        col1.markdown('<div class="metric-box"><div class="metric-title">Recall</div><div class="metric-val">81.0%</div></div>', unsafe_allow_html=True)
        col2.markdown('<div class="metric-box"><div class="metric-title">F1-Score</div><div class="metric-val">83.8%</div></div>', unsafe_allow_html=True)

    artifacts = load_artifacts()
    model, scaler, label_encoder = artifacts

    st.markdown("""
    ### 📤 Sube una imagen de la lesión de piel
    Carga una imagen en formato **JPG, PNG o JPEG** para que el modelo realice el diagnóstico.
    """)

    uploaded_file = st.file_uploader(
        "Selecciona la imagen",
        type=None,
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        try:
            img, img_resized, X_scaled = preprocess_image(uploaded_file, {"scaler": scaler})
        except ValueError as e:
            st.error(str(e))
            st.stop()

        col_preview, col_result = st.columns([1, 1], gap="large")

        with col_preview:
            st.markdown("#### 🖼️ Imagen Cargada")
            st.image(img, use_container_width=True, caption="Vista original de la lesión")

        with col_result:
            st.markdown("#### 🧠 Predicción del Modelo")
            with st.spinner("Analizando la lesión..."):
                y_pred = model.predict(X_scaled)[0]
                proba = model.predict_proba(X_scaled)[0]

            prob_benign = proba[0]
            prob_malignant = proba[1]
            pred_class = label_encoder.inverse_transform([y_pred])[0]

            if pred_class == "Malignant":
                st.markdown(f"""
                <div class="pred-card maligno">
                    <div style="font-size:1rem; opacity:0.9;">DIAGNÓSTICO:</div>
                    <div class="pred-label">⚠️ MALIGNO</div>
                    <div class="pred-desc">Lesión detectada como melanoma sospechoso</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                <div class="conf-grid">
                    <div class="conf-cell"><div class="conf-value">{:.1f}%</div><div class="conf-label">Prob. Maligno</div></div>
                    <div class="conf-cell"><div class="conf-value">{:.1f}%</div><div class="conf-label">Prob. Benigno</div></div>
                </div>
                """.format(prob_malignant * 100, prob_benign * 100), unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-card benigno">
                    <div style="font-size:1rem; opacity:0.9;">DIAGNÓSTICO:</div>
                    <div class="pred-label">✅ BENIGNO</div>
                    <div class="pred-desc">Lesión clasificada como no cancerosa</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                <div class="conf-grid">
                    <div class="conf-cell"><div class="conf-value">{:.1f}%</div><div class="conf-label">Prob. Benigno</div></div>
                    <div class="conf-cell"><div class="conf-value">{:.1f}%</div><div class="conf-label">Prob. Maligno</div></div>
                </div>
                """.format(prob_benign * 100, prob_malignant * 100), unsafe_allow_html=True)

            st.markdown("#### 📊 Nivel de Confianza")
            if prob_malignant > prob_benign:
                st.progress(int(prob_malignant * 100))
                st.caption(f"Confianza del diagnóstico: {prob_malignant:.1%}")
            else:
                st.progress(int(prob_benign * 100))
                st.caption(f"Confianza del diagnóstico: {prob_benign:.1%}")

        st.info("""
        **⚠️ Aviso importante:** Este sistema es una herramienta educativa de apoyo al diagnóstico. **No reemplaza la evaluación de un dermatólogo profesional.** Ante cualquier sospecha, consulta a un especialista.
        """, icon="💡")

    else:
        st.markdown('<div class="upload-area"><div style="font-size: 3.5rem;">🩻</div><br/><b>Sube tu imagen aquí</b><br/><span style="color:#b8b8d1; font-size:0.9rem;">Arrastra y suelta una imagen para comenzar el análisis</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="footer">Melanoma Detector AI · Proyecto de Machine Learning · Regresión Logística</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
