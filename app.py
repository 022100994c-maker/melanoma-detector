import os
import numpy as np
import joblib
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

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


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def plot_sigmoid(model, z_highlight=None, prob_highlight=None):
    z = np.linspace(-8, 8, 300)
    s = sigmoid(z)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=110, facecolor='none')
    ax.plot(z, s, color='#00d2ff', linewidth=3, label='σ(z) = 1/(1+e⁻ᶻ)')
    ax.axhline(0.5, color='#b8b8d1', linestyle='--', linewidth=1, label='Umbral = 0.5')
    ax.axvline(0, color='#b8b8d1', linestyle=':', linewidth=1, alpha=0.7)
    ax.axhline(0, color='#b8b8d1', linestyle='-', linewidth=0.5, alpha=0.4)
    ax.axhline(1, color='#b8b8d1', linestyle='-', linewidth=0.5, alpha=0.4)
    ax.set_xlabel('z = β₀ + β₁x₁ + ... + βₙxₙ (score lineal)', color='#d0d0e8')
    ax.set_ylabel('Probabilidad P(y=1)', color='#d0d0e8')
    ax.set_title('Función Sigmoide aplicada al modelo', color='#ffffff', fontweight='bold')
    ax.tick_params(colors='#d0d0e8')
    for spine in ax.spines.values():
        spine.set_color('#555577')
    ax.grid(True, alpha=0.15)
    ax.legend(loc='center right', facecolor='#24243e', edgecolor='#555577', labelcolor='#ffffff')

    if z_highlight is not None and prob_highlight is not None:
        ax.scatter([z_highlight], [prob_highlight], color='#ff6b6b', s=90, zorder=5, label=f'Predicción actual (prob = {prob_highlight:.2f})')
        ax.annotate(f'z = {z_highlight:.2f}\nP = {prob_highlight:.2f}', xy=(z_highlight, prob_highlight),
                    xytext=(z_highlight + 1.2, prob_highlight - 0.15),
                    color='#ffffff', fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='#ff6b6b'))
        ax.legend(loc='center right', facecolor='#24243e', edgecolor='#555577', labelcolor='#ffffff')

    fig.tight_layout()
    return fig


def compute_z(model, X_scaled):
    coef = model.coef_[0]
    intercept = model.intercept_[0]
    return float(np.dot(X_scaled[0], coef) + intercept)


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

        .formula-box {
            background: rgba(255,255,255,0.06);
            border-left: 4px solid #00d2ff;
            border-radius: 8px;
            padding: 1rem 1.5rem;
            margin: 1rem 0;
            font-family: 'Courier New', monospace;
            font-size: 1.1rem;
            color: #e0e0ff;
            text-align: center;
        }

        .edu-card {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
            margin: 0.6rem 0;
        }

        .edu-card h4 {
            color: #00d2ff;
            margin-top: 0;
            margin-bottom: 0.4rem;
        }

        .edu-card p, .edu-card li {
            color: #d0d0e8;
            font-size: 0.95rem;
        }

        .badge-sigmoid {
            display: inline-block;
            background: rgba(0,210,255,0.15);
            color: #00d2ff;
            border-radius: 20px;
            padding: 0.2rem 0.9rem;
            font-weight: 600;
            font-size: 0.85rem;
        }

        .hide-default {display: none !important;}

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

    tab_pred, tab_edu = st.tabs(["🔍 Predicción", "📚 ¿Cómo funciona? (Sigmoide)"])

    with tab_pred:
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

            y_pred = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]
            st.session_state["last_result"] = {
                "z": compute_z(model, X_scaled),
                "prob_malignant": float(proba[1])
            }

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

    with tab_edu:
        st.markdown('<span class="badge-sigmoid">🧠 Fundamentos del modelo</span>', unsafe_allow_html=True)
        st.markdown("### ¿Cómo decide el modelo? La función Sigmoide")

        st.markdown("""
        La **Regresión Logística** no predice directamente la clase, sino la **probabilidad** de que la lesión sea maligna (clase 1). Para ello usa la **función sigmoide** (también llamada función logística):
        """)

        st.markdown('<div class="formula-box">σ(z) = 1 / (1 + e⁻ᶻ)</div>', unsafe_allow_html=True)

        st.markdown("""
        donde **z** es una *combinación lineal* de los 3,072 píxeles de la imagen (características):

        <div class="formula-box">z = β₀ + β₁·x₁ + β₂·x₂ + ... + β₃₀₇₂·x₃₀₇₂</div>

        - Los **β** (betas) son los *coeficientes* que el modelo aprendió durante el entrenamiento.
        - Cada **x** es un píxel de la imagen (ya normalizado con el scaler).
        - La sigmoide convierte cualquier valor de **z** (de −∞ a +∞) en una **probabilidad entre 0 y 1**.
        """, unsafe_allow_html=True)

        st.markdown("### 📈 Interpretación de la curva")
        st.markdown("""
        <div class="edu-card">
        <h4>📌 ¿Cómo se interpreta?</h4>
        <ul>
            <li>Cuando <strong>z → +∞</strong>, la sigmoide se acerca a <strong>1</strong> (alta probabilidad de maligno).</li>
            <li>Cuando <strong>z → −∞</strong>, la sigmoide se acerca a <strong>0</strong> (alta probabilidad de benigno).</li>
            <li>En <strong>z = 0</strong>, la probabilidad es exactamente <strong>0.5</strong> (punto de indecisión).</li>
            <li>El modelo clasifica como <strong>Maligno</strong> si P(y=1) ≥ 0.5, y como <strong>Benigno</strong> en caso contrario.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        if "last_result" in st.session_state:
            last = st.session_state["last_result"]
            st.markdown("### 🖼️ Tu imagen en la curva")
            st.markdown("""
            <div class="edu-card">
            <h4>📍 Posición de tu última predicción</h4>
            <p>El punto <strong>rojo</strong> en la curva muestra dónde quedó tu imagen según su score lineal <strong>z</strong>. La proyección vertical indica la probabilidad de maligno que el modelo le asignó. Este es exactamente el valor que ves como porcentaje en la pestaña de <strong>Predicción</strong>.</p>
            </div>
            """, unsafe_allow_html=True)
            fig_own = plot_sigmoid(model, z_highlight=last["z"], prob_highlight=last["prob_malignant"])
            st.pyplot(fig_own)
            pc1, pc2 = st.columns(2)
            pc1.metric("z de tu imagen", f"{last['z']:.2f}")
            pc2.metric("P(maligno) de tu imagen", f"{last['prob_malignant']:.3f}")
            st.divider()
            st.markdown("### 🧪 Prueba interactiva manual")

        else:
            st.markdown("### 🧪 Prueba interactiva manual")

        st.markdown("""
        <div class="edu-card">
        <h4>🔍 Interacción</h4>
        <p>Ajusta el valor de <strong>z</strong> (el score lineal) con el deslizador y observa cómo la curva convierte ese valor en una probabilidad. Así funciona internamente cada predicción que hace el modelo con tus imágenes.</p>
        </div>
        """, unsafe_allow_html=True)

        z_interactive = st.slider("Valor de z (score lineal)", -8.0, 8.0, 0.0, 0.1)
        prob_interactive = float(sigmoid(z_interactive))
        fig_inter = plot_sigmoid(model, z_highlight=z_interactive, prob_highlight=prob_interactive)
        st.pyplot(fig_inter)

        c1, c2 = st.columns(2)
        c1.metric("z (lineal)", f"{z_interactive:.2f}")
        c2.metric("Probabilidad P(maligno)", f"{prob_interactive:.3f}")

        if prob_interactive >= 0.5:
            st.success(f"Clasificación: **MALIGNO** (p = {prob_interactive:.2f} ≥ 0.5)")
        else:
            st.success(f"Clasificación: **BENIGNO** (p = {prob_interactive:.2f} < 0.5)")

    st.markdown('<div class="footer">Melanoma Detector AI · Proyecto de Machine Learning · Regresión Logística</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
