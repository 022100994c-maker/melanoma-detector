import os
from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
OUTPUT_PDF = os.path.join(PROJECT_DIR, "docs", "Informe_Melanoma_Detector.pdf")

INTEGRANTES = [
    ("Suárez Condori Juan Gabriel", "022100.."),
    ("Ramos Ticahuanca Gianella Alexandra", "022100.."),
    ("Jesus del Aguila Garcia", "022100.."),
    ("Mendoza Torres Lincol Jhon", "022100.."),
]

print("Proyecto:", PROJECT_DIR)

STYLES = getSampleStyleSheet()

def style(name, **kw):
    base = kw.pop("base", "BodyText")
    s = ParagraphStyle(name, parent=STYLES[base], **kw)
    return s

TITLE = style("Title", fontName="Helvetica-Bold", fontSize=22, leading=26, alignment=TA_CENTER, textColor=colors.HexColor("#0f3a6b"))
SUBTITLE = style("Subtitle", fontName="Helvetica", fontSize=13, leading=18, alignment=TA_CENTER, textColor=colors.HexColor("#333333"))
H1 = style("H1", base="Heading1", fontName="Helvetica-Bold", fontSize=15, leading=19, textColor=colors.HexColor("#0f3a6b"), spaceBefore=16, spaceAfter=8)
H2 = style("H2", base="Heading2", fontName="Helvetica-Bold", fontSize=12.5, leading=16, textColor=colors.HexColor("#1f5fa8"), spaceBefore=10, spaceAfter=6)
BODY = style("Body", base="BodyText", fontName="Helvetica", fontSize=10.5, leading=15, alignment=TA_JUSTIFY, textColor=colors.HexColor("#1a1a1a"))
BULLET = style("Bullet", base="BodyText", fontName="Helvetica", fontSize=10.5, leading=15, leftIndent=16, bulletIndent=6, textColor=colors.HexColor("#1a1a1a"))
CAPTION = style("Caption", base="BodyText", fontName="Helvetica-Oblique", fontSize=9, leading=12, alignment=TA_CENTER, textColor=colors.HexColor("#555555"), spaceBefore=4, spaceAfter=12)
CENTER = style("Center", base="BodyText", alignment=TA_CENTER, fontName="Helvetica", fontSize=10.5, leading=15)


def build_pdf():
    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=A4,
        rightMargin=2.2*cm,
        leftMargin=2.2*cm,
        topMargin=2.0*cm,
        bottomMargin=2.0*cm,
        title="Informe - Clasificacion de Melanoma con Regresion Logistica",
        author="Equipo Melanoma Detector AI",
    )

    story = []

    # ------------------- CARATULA -------------------
    story.append(Spacer(1, 2.5*cm))
    story.append(Paragraph("TALLER 1.2: MODELOS DE REGRESIÓN LOGÍSTICA", SUBTITLE))
    story.append(Spacer(1, 1.2*cm))
    story.append(Paragraph("Clasificación Binaria con Regresión Logística y Despliegue de Modelo", style("CoverSub", base="Title", fontSize=18, leading=24, alignment=TA_CENTER, textColor=colors.HexColor("#1f5fa8"))))
    story.append(Spacer(1, 1.0*cm))
    story.append(Paragraph("🩺 Melanoma Detector AI", TITLE))
    story.append(Paragraph("Detección de Melanoma: Benigno vs Maligno", style("CoverSub2", base="Title", fontSize=14, leading=20, alignment=TA_CENTER, textColor=colors.HexColor("#555555"))))
    story.append(Spacer(1, 2.5*cm))

    story.append(Paragraph("<b>INTEGRANTES DEL EQUIPO</b>", style("Center", base="BodyText", fontName="Helvetica-Bold", fontSize=12)))
    story.append(Spacer(1, 0.4*cm))

    integ_data = [[Paragraph("<b>Nombres y Apellidos</b>", style("Cell", base="BodyText", fontName="Helvetica-Bold", fontSize=10)), Paragraph("<b>Código Universitario</b>", style("Cell", base="BodyText", fontName="Helvetica-Bold", fontSize=10))]]
    for nombre, cod in INTEGRANTES:
        integ_data.append([Paragraph(nombre, style("Cell", base="BodyText", fontSize=10)), Paragraph(cod, style("Cell", base="BodyText", fontSize=10))])

    integ_table = Table(integ_data, colWidths=[12*cm, 5.5*cm])
    integ_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0f3a6b")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.6, colors.HexColor("#888888")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#eef3fa")]),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
    ]))
    story.append(integ_table)
    story.append(Spacer(1, 1.8*cm))

    story.append(Paragraph("Fecha: 02 de septiembre de 2026", CENTER))
    story.append(Paragraph("Curso: Inteligencia Artificial", CENTER))
    story.append(PageBreak())

    # ------------------- RESUMEN EJECUTIVO -------------------
    story.append(Paragraph("Resumen Ejecutivo", H1))
    story.append(Paragraph(
        "El presente trabajo tiene como objetivo implementar un modelo de clasificación binaria mediante "
        "<b>Regresión Logística</b> para el diagnóstico de lesiones de piel, discriminando entre dos clases "
        "mutuamente excluyentes: <b>Benigno (no canceroso)</b> y <b>Maligno (melanoma)</b>. Se utilizó un dataset "
        "de <b>14,879 imágenes</b> de lesiones dérmicas, divididas en conjunto de entrenamiento (11,879) y prueba (2,000). "
        "A partir de cada imagen se extrajeron las características de píxeles redimensionadas a 32x32 RGB (3,072 variables), "
        "se normalizaron con StandardScaler y se entrenó un clasificador de regresión logística.", BODY))
    story.append(Paragraph(
        "Los resultados obtenidos sobre el conjunto de prueba fueron: <b>Accuracy 84.3%</b>, <b>Precision 86.7%</b>, "
        "<b>Recall 81.0%</b>, <b>F1-Score 83.8%</b> y <b>AUC-ROC 91.97%</b>. Asimismo, se desarrolló y desplegó un "
        "aplicativo web interactivo en Streamlit que permite al usuario cargar una imagen y obtener la predicción "
        "(Benigno/Maligno) junto con la probabilidad asignada por el modelo.", BODY))
    story.append(Paragraph(
        "El modelo logró un rendimiento aceptable para un clasificador lineal sobre características de píxeles, "
        "demostrando la viabilidad de la regresión logística como línea base para el diagnóstico asistido de melanoma.", BODY))

    # ------------------- DEFINICION DEL PROBLEMA -------------------
    story.append(Paragraph("Definición del Problema", H1))
    story.append(Paragraph("Contexto", H2))
    story.append(Paragraph(
        "El melanoma es uno de los tipos de cáncer de piel más agresivos, pero su detección temprana incrementa "
        "significativamente la tasa de supervivencia. La inspección visual de lesiones dérmicas (lunares, manchas) "
        "para distinguir entre lesiones benignas y malignas es una tarea compleja y subjetiva que requiere epecialistas "
        "capacitados. En este contexto, un sistema de apoyo al diagnóstico basado en aprendizaje de máquina puede "
        "asistir a los profesionales de la salud y a la población en general.", BODY))

    story.append(Paragraph("Clases Binarias del Problema", H2))
    story.append(Paragraph(
        "Se plantea un problema de <b>clasificación binaria</b> con dos clases mutuamente excluyentes:", BODY))
    story.append(Paragraph("• <b>Clase 0 – Benigno:</b> lesión no cancerosa.", BULLET))
    story.append(Paragraph("• <b>Clase 1 – Maligno:</b> lesión cancerosa (melanoma).", BULLET))

    story.append(Paragraph("Valor del Negocio / Aplicación", H2))
    story.append(Paragraph(
        "El valor de la aplicación radica en ofrecer un primer tamizaje (screening) automático y de bajo costo. "
        "Un sistema como este podría priorizar las lesiones que requieren consulta inmediata con un dermatólogo, "
        "reduciendo los tiempos de diagnóstico y apoyando a poblaciones con acceso limitado a especialistas. Se "
        "enfatiza que constituye una herramienta de <b>apoyo educativo</b> y no reemplaza el juicio profesional.", BODY))

    # ------------------- EDA Y PREPARACION -------------------
    story.append(Paragraph("Análisis Exploratorio y Preparación de Datos", H1))

    story.append(Paragraph("Carga y Exploración de Datos (EDA)", H2))
    story.append(Paragraph(
        "El dataset se encontraba organizado en carpetas por clase y por conjunto (train/test). La distribución "
        "de las observaciones fue la siguiente:", BODY))

    data_rows = [
        ["Conjunto", "Benignos", "Malignos", "Total"],
        ["Entrenamiento", "6,289", "5,590", "11,879"],
        ["Prueba", "1,000", "1,000", "2,000"],
        ["Total", "7,289", "6,590", "13,879"],
    ]
    data_table = Table(data_rows, colWidths=[4.5*cm, 3*cm, 3*cm, 3*cm])
    data_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0f3a6b")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.6, colors.HexColor("#888888")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#eef3fa")]),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("ALIGN", (1,0), (-1,-1), "CENTER"),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(data_table)
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "El conjunto está <b>balanceado</b> (proporción de ~54% benignos y ~46% malignos en entrenamiento), lo cual "
        "es favorable para la clasificación sin necesidad de técnicas de balanceo de clases.", BODY))

    story.append(Paragraph(
        "Los gráficos del análisis exploratorio incluyen la distribución de clases, muestras de imágenes por clase, "
        "la distribución de la intensidad promedio de píxeles y las imágenes promedio de cada clase. El análisis "
        "permite observar diferencias sutiles en la textura y tonalidad entre ambas clases, que son capturadas por "
        "el modelo a través de los píxeles.", BODY))

    eda_img = os.path.join(PROJECT_DIR, "plots", "eda_analysis.png")
    if os.path.exists(eda_img):
        story.append(Spacer(1, 0.3*cm))
        story.append(Image(eda_img, width=17*cm, height=10.3*cm))
        story.append(Paragraph("Figura 1. Análisis Exploratorio de Datos (EDA).", CAPTION))

    story.append(Paragraph("Preprocesamiento de Datos", H2))
    story.append(Paragraph(
        "No se presentaron valores faltantes, ya que todas las imágenes se encontraban completas. Las etapas de "
        "preprocesamiento e ingeniería de atributos fueron:", BODY))
    story.append(Paragraph("• <b>Redimensionamiento:</b> cada imagen de 224x224 se redimensionó a 32x32 para reducir la dimensionalidad y el costo computacional.", BULLET))
    story.append(Paragraph("• <b>Aplanamiento:</b> cada imagen 32x32x3 se convirtió en un vector unidimensional de 3,072 características (píxeles).", BULLET))
    story.append(Paragraph("• <b>Codificación de etiquetas:</b> se usó LabelEncoder (Benign → 0, Malign → 1).", BULLET))
    story.append(Paragraph("• <b>Estandarización:</b> se aplicó StandardScaler para normalizar las características a media 0 y desviación estándar 1.", BULLET))
    story.append(Paragraph("• <b>División del dataset:</b> se mantuvo la separación original train/test (80/20) para entrenamiento y evaluación imparcial.", BULLET))

    # ------------------- MODELADO -------------------
    story.append(Paragraph("Modelado con Regresión Logística", H1))

    story.append(Paragraph("Justificación del Modelo", H2))
    story.append(Paragraph(
        "La regresión logística es un modelo de clasificación lineal ampliamente utilizado por su interpretabilidad, "
        "su bajo costo computacional y su capacidad de producir probabilidades de pertenencia a cada clase. Para un "
        "problema binario como la detección de melanoma, resulta una elección adecuada como modelo base y permite "
        "analizar directamente la influencia de cada característica a través de sus coeficientes.", BODY))

    story.append(Paragraph("La Función Sigmoide", H2))
    story.append(Paragraph(
        "La regresión logística modela la probabilidad P(y=1|x) mediante la <b>función sigmoide (función logística)</b>:", BODY))
    story.append(Paragraph(
        "σ(z) = 1 / (1 + e^(−z))", style("Formula", base="BodyText", alignment=TA_CENTER, fontName="Courier-Bold", fontSize=13, spaceBefore=6, spaceAfter=6, textColor=colors.HexColor("#0f3a6b"))))
    story.append(Paragraph(
        "donde z es la combinación lineal de las características: z = β₀ + β₁x₁ + β₂x₂ + ... + β₃₀₇₂x₃₀₇₂. "
        "La sigmoide transforma cualquier valor real de z en un valor entre 0 y 1, interpretable como probabilidad. "
        "El punto z=0 corresponde a una probabilidad de 0.5 (umbral de decisión): si P ≥ 0.5 se clasifica como "
        "Maligno; en caso contrario, Benigno.", BODY))

    story.append(Paragraph("Configuración de Hiperparámetros", H2))
    story.append(Paragraph(
        "El modelo se configuró con los siguientes hiperparámetros de scikit-learn:", BODY))
    hp_rows = [
        ["Parámetro", "Valor"],
        ["Modelo", "LogisticRegression()"],
        ["solver", "lbfgs"],
        ["max_iter", "1000"],
        ["C (regularización)", "1.0"],
        ["random_state", "42"],
        ["n_jobs", "-1"],
        ["Escalado", "StandardScaler"],
    ]
    hp_table = Table(hp_rows, colWidths=[6*cm, 9.5*cm])
    hp_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0f3a6b")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.6, colors.HexColor("#888888")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#eef3fa")]),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(hp_table)
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Matriz de Coeficientes / Pesos", H2))
    story.append(Paragraph(
        "El modelo aprendió un vector de 3,072 coeficientes (uno por píxel) más un intercepto. El análisis de los "
        "coeficientes permite identificar qué regiones de la imagen influyen más en la decisión. La visualización "
        "de los pesos reorganizados espacialmente muestra los píxeles de mayor relevancia en la clasificación.", BODY))

    coeff_img = os.path.join(PROJECT_DIR, "plots", "coefficients.png")
    if os.path.exists(coeff_img):
        story.append(Spacer(1, 0.3*cm))
        story.append(Image(coeff_img, width=14*cm, height=5.8*cm))
        story.append(Paragraph("Figura 2. Análisis de coeficientes del modelo por píxel.", CAPTION))

    # ------------------- RESULTADOS -------------------
    story.append(Paragraph("Resultados y Discusión de Métricas", H1))
    story.append(Paragraph(
        "A continuación se presentan los resultados obtenidos sobre el conjunto de <b>prueba</b> (2,000 imágenes).", BODY))

    story.append(Paragraph("Matriz de Confusión", H2))
    cm_img = os.path.join(PROJECT_DIR, "plots", "evaluation.png")
    if os.path.exists(cm_img):
        story.append(Image(cm_img, width=17*cm, height=6.1*cm))
        story.append(Paragraph("Figura 3. Matriz de confusión y curva ROC del modelo.", CAPTION))

    story.append(Paragraph(
        "La matriz de confusión desglosa los aciertos y errores del modelo. Se observa un buen equilibrio entre "
        "los falsos negativos y falsos positivos, lo cual es relevante en el ámbito médico donde las consecuencias "
        "de ambos tipos de error difieren.", BODY))

    story.append(Paragraph("Tabla Comparativa de Métricas", H2))
    metric_rows = [
        ["Métrica", "Valor", "Interpretación"],
        ["Exactitud (Accuracy)", "84.30%", "Proporción total de predicciones correctas."],
        ["Precisión (Precision)", "86.72%", "De los predichos malignos, % que realmente lo son."],
        ["Sensibilidad (Recall)", "81.00%", "De los malignos reales, % detectados."],
        ["F1-Score", "83.76%", "Media armónica entre precisión y recall."],
        ["AUC-ROC", "91.97%", "Capacidad de discriminar entre clases."],
    ]
    metric_table = Table(metric_rows, colWidths=[5.5*cm, 3.2*cm, 7*cm])
    metric_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0f3a6b")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.6, colors.HexColor("#888888")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#eef3fa")]),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("ALIGN", (1,1), (1,-1), "CENTER"),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(metric_table)
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Interpretación Clínica / Operativa de los Errores", H2))
    story.append(Paragraph(
        "En el contexto de diagnóstico médico, los errores tienen consecuencias asimétricas. Un <b>falso negativo</b> "
        "(clasificar como benigno un melanoma real) es el error más peligroso, ya que puede retrasar el tratamiento. "
        "El modelo presenta un recall del 81%, lo que significa que detecta correctamente el 81% de los melanomas reales. "
        "Un <b>falso positivo</b> (clasificar como maligno una lesión benigna) genera preocupación y consultas adicionales, "
        "pero es menos riesgoso desde el punto de vista clínico.", BODY))
    story.append(Paragraph(
        "El AUC de 91.97% indica una buena capacidad general para separar ambas clases. Para mejorar aún más la "
        "sensibilidad (prioridad clínica), se podría ajustar el umbral de decisión por debajo de 0.5, a costa de "
        "aumentar los falsos positivos.", BODY))

    # ------------------- MANUAL Y ARQUITECTURA -------------------
    story.append(Paragraph("Manual de Usuario y Arquitectura del Aplicativo", H1))

    story.append(Paragraph("Manual de Usuario", H2))
    story.append(Paragraph("Pasos para utilizar la aplicación web:", BODY))
    story.append(Paragraph("1. Acceder a la URL pública de la aplicación (Streamlit Community Cloud).", BULLET))
    story.append(Paragraph("2. En la pestaña <b>“Predicción”</b>, hacer clic en el área de carga y seleccionar (o arrastrar) una imagen de lesión de piel en formato JPG, JPEG o PNG.", BULLET))
    story.append(Paragraph("3. El sistema muestra la imagen cargada y el resultado con la predicción: <b>MALIGNO</b> (tarjeta roja) o <b>BENIGNO</b> (tarjeta verde).", BULLET))
    story.append(Paragraph("4. Se muestran las probabilidades de cada clase y el nivel de confianza del diagnóstico.", BULLET))
    story.append(Paragraph("5. En la pestaña <b>“¿Cómo funciona? (Sigmoide)”</b> se presenta la explicación interactiva del modelo con la curva sigmoide y la posición de la última predicción.", BULLET))

    story.append(Paragraph(
        "URL pública del aplicativo: <b>https://melanoma-detector.streamlit.app</b> (desplegado en Streamlit Community Cloud).", BODY))

    story.append(Paragraph("Arquitectura y Flujo de Datos", H2))
    story.append(Paragraph(
        "La arquitectura de la solución sigue el siguiente flujo de datos hacia la aplicación:", BODY))
    story.append(Paragraph(
        "1. <b>Entrenamiento:</b> las imágenes de train se procesan (redimensionado y aplanado), se escalan y se usan "
        "para entrenar la regresión logística. El modelo, el scaler y el encoder se guardan como archivos .pkl "
        "(joblib).", BULLET))
    story.append(Paragraph(
        "2. <b>Inferencia (App):</b> el usuario carga una imagen → se redimensiona a 32x32 → se aplanan los píxeles → "
        "se normalizan con el scaler guardado → el modelo calcula la combinación lineal z → la sigmoide produce la "
        "probabilidad → se clasifica según el umbral 0.5 → se muestra el resultado.", BULLET))
    story.append(Paragraph(
        "3. <b>Despliegue:</b> la aplicación Streamlit se aloja en la nube y expone la interfaz al usuario final vía navegador.", BULLET))

    story.append(Paragraph("Diagrama simple del flujo de datos:", BODY))
    flow_rows = [
        ["Usuario/Imagen", "Preprocesamiento", "Modelo (z y σ)", "Resultado"],
        ["Carga de imagen JPG/PNG", "Resize 32x32 + flatten + StandardScaler", "Regresión Logística + Sigmoide", "Benigno/Maligno + Probabilidad"],
    ]
    flow_table = Table(flow_rows, colWidths=[4.2*cm, 4.4*cm, 4.2*cm, 3.9*cm])
    flow_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0f3a6b")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.6, colors.HexColor("#888888")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#eef3fa")]),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
    ]))
    story.append(flow_table)

    # ------------------- CONCLUSIONES -------------------
    story.append(Paragraph("Conclusiones y Recomendaciones", H1))
    story.append(Paragraph("Conclusiones", H2))
    story.append(Paragraph("• Se implementó un modelo de regresión logística capaz de clasificar lesiones de piel en benignas y malignas con un 84.3% de exactitud y un AUC-ROC de 91.97%.", BULLET))
    story.append(Paragraph("• El análisis de coeficientes permitió interpretar qué características (píxeles) influyen más en la decisión del modelo.", BULLET))
    story.append(Paragraph("• Se completó el ciclo completo de Machine Learning, desde el análisis del problema hasta el despliegue de una aplicación web funcional e interactiva.", BULLET))
    story.append(Paragraph("• La aplicación proporciona la predicción (0/1) junto con la probabilidad asignada, cumpliendo el requisito funcional del despliegue.", BULLET))

    story.append(Paragraph("Recomendaciones y Mejoras Futuras", H2))
    story.append(Paragraph("• Explorar aumentación de datos para mejorar la generalización del modelo.", BULLET))
    story.append(Paragraph("• Probar arquitecturas de redes neuronales convolucionales (CNN) que capturen mejor las características espaciales y de textura de las lesiones.", BULLET))
    story.append(Paragraph("• Ajustar el umbral de decisión para priorizar la sensibilidad (detección de malignos) en el ámbito clínico.", BULLET))
    story.append(Paragraph("• Incorporar visualizaciones de interpretabilidad como mapas de activación para explicar cada predicción.", BULLET))
    story.append(Paragraph("• Ampliar el dataset con más variedad para mejorar la robustez y reducir los falsos negativos.", BULLET))

    # ------------------- ANEXOS -------------------
    story.append(Paragraph("Anexos", H1))
    story.append(Paragraph("Repositorio de Código", H2))
    story.append(Paragraph(
        "El código fuente del proyecto está disponible en el repositorio de GitHub:<br/>"
        "<link href='https://github.com/022100994c-maker/melanoma-detector'><b>"
        "https://github.com/022100994c-maker/melanoma-detector</b></link><br/><br/>"
        "El repositorio contiene los scripts de entrenamiento (train_model.py), la aplicación web (app.py), "
        "el modelo entrenado (archivos .pkl) y la documentación.", BODY))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph(
        "** Nota: los códigos universitarios de los integrantes deben ser completados con los valores reales.", style("Note", base="BodyText", fontSize=8.5, textColor=colors.HexColor("#888888"))))

    doc.build(story)
    print(f"PDF generado con éxito: {OUTPUT_PDF}")


if __name__ == "__main__":
    build_pdf()
