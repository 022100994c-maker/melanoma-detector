# 🩺 Melanoma Detector AI

Clasificación binaria de lesiones de piel (**Benigno** vs **Maligno**) mediante **Regresión Logística**, con un aplicativo web interactivo construido en **Streamlit**.

## 🚀 Demo / Deploy

App desplegada en Streamlit Community Cloud: *(añadir URL después del deploy)*

## 📋 Descripción del Proyecto

Este proyecto aplica **Machine Learning** para discriminar entre lesiones de piel benignas y malignas (melanoma) a partir de imágenes. Cumple el ciclo completo de un proyecto de ML:

1. Análisis exploratorio de datos (EDA)
2. Preprocesamiento y normalización de características
3. Entrenamiento con Regresión Logística
4. Evaluación con métricas estándar
5. Despliegue con interfaz web interactiva

## 🗂️ Estructura del Repositorio

```
melanoma-detector/
├── app.py                 # Aplicación web Streamlit
├── train_model.py         # Script de entrenamiento y evaluación
├── requirements.txt       # Dependencias del proyecto
├── model/                 # Modelo, scaler y label encoder exportados
│   ├── melanoma_logistic_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── model_config.pkl
├── plots/                 # Gráficos EDA, coeficientes y evaluación
└── data/                  # Dataset de imágenes (train/test)
```

## 📊 Resultados del Modelo (Conjunto de Test)

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 84.3% |
| **Precision** | 86.7% |
| **Recall (Sensitivity)** | 81.0% |
| **F1-Score** | 83.8% |
| **AUC-ROC** | 92.0% |

## 🧠 Sobre el Modelo

- **Características:** Píxeles de imágenes redimensionadas a 32×32 RGB (vector de 3,072 features).
- **Preprocesamiento:** `StandardScaler` para normalización.
- **Modelo:** `LogisticRegression` de scikit-learn con `solver='lbfgs'`.
- **Etiquetas:** Benign = 0, Malignant = 1.

## 🖥️ Ejecución Local

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. (Opcional) Entrenar el modelo

```bash
python train_model.py
```

### 3. Ejecutar la aplicación

```bash
streamlit run app.py
```

Abre el navegador en `http://localhost:8501` y sube una imagen de una lesión de piel.

## ☁️ Despliegue en Streamlit Community Cloud

1. Sube este repositorio a GitHub.
2. Ve a [share.streamlit.io](https://share.streamlit.io) e inicia sesión con tu cuenta de GitHub.
3. Haz clic en **"New app"**.
4. Selecciona el repositorio y el archivo principal (`app.py`).
5. Haz clic en **Deploy**. ¡Listo!

## ⚠️ Aviso

Esta herramienta es **educativa y de apoyo al diagnóstico**. **No reemplaza la evaluación de un dermatólogo profesional.** Ante cualquier sospecha de melanoma, consulta a un especialista.

## 👥 Autores

Proyecto académico de clasificación binaria con Regresión Logística.

## 📄 Licencia

Uso educativo.
