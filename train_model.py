import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, roc_auc_score,
    classification_report
)
import joblib
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
MODEL_DIR = os.path.join(BASE_DIR, "model")
IMG_SIZE = 32


def load_images_from_folder(folder_path, label):
    images = []
    labels = []
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for fname in files:
        img_path = os.path.join(folder_path, fname)
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img_array = np.array(img).flatten()
            images.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"  Error cargando {img_path}: {e}")
    return images, labels


def load_dataset():
    print("=" * 60)
    print("FASE I: CARGA DE DATOS")
    print("=" * 60)

    train_benign_X, train_benign_y = load_images_from_folder(os.path.join(TRAIN_DIR, "Benign"), "Benign")
    train_malignant_X, train_malignant_y = load_images_from_folder(os.path.join(TRAIN_DIR, "Malignant"), "Malignant")
    test_benign_X, test_benign_y = load_images_from_folder(os.path.join(TEST_DIR, "Benign"), "Benign")
    test_malignant_X, test_malignant_y = load_images_from_folder(os.path.join(TEST_DIR, "Malignant"), "Malignant")

    X_train = np.array(train_benign_X + train_malignant_X)
    y_train = np.array(train_benign_y + train_malignant_y)
    X_test = np.array(test_benign_X + test_malignant_X)
    y_test = np.array(test_benign_y + test_malignant_y)

    print(f"Train: {len(X_train)} imágenes ({len(train_benign_X)} Benign, {len(train_malignant_X)} Malignant)")
    print(f"Test:  {len(X_test)} imágenes ({len(test_benign_X)} Benign, {len(test_malignant_X)} Malignant)")
    print(f"Tamaño de feature vector: {X_train.shape[1]} ({IMG_SIZE}x{IMG_SIZE}x3)")

    return X_train, y_train, X_test, y_test


def perform_eda(X_train, y_train):
    print("\n" + "=" * 60)
    print("FASE II: ANÁLISIS EXPLORATORIO (EDA)")
    print("=" * 60)

    os.makedirs(os.path.join(BASE_DIR, "plots"), exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Análisis Exploratorio de Datos - Melanoma Classification", fontsize=16, fontweight='bold')

    counts = pd.Series(y_train).value_counts()
    colors = ['#2ecc71', '#e74c3c']
    axes[0, 0].bar(counts.index, counts.values, color=colors, edgecolor='black', linewidth=0.5)
    axes[0, 0].set_title("Distribución de Clases (Train)", fontweight='bold')
    axes[0, 0].set_ylabel("Cantidad")
    for i, v in enumerate(counts.values):
        axes[0, 0].text(i, v + 50, str(v), ha='center', fontweight='bold', fontsize=12)

    benign_idx = np.where(y_train == "Benign")[0]
    malignant_idx = np.where(y_train == "Malignant")[0]
    for idx, (i, label) in enumerate([(benign_idx[0], "Benign (Muestra)"), (malignant_idx[0], "Malignant (Muestra)")]):
        img = X_train[i].reshape(IMG_SIZE, IMG_SIZE, 3)
        axes[0, idx + 1].imshow(img.astype(np.uint8))
        axes[0, idx + 1].set_title(label, fontweight='bold')
        axes[0, idx + 1].axis('off')

    pixel_means_benign = X_train[benign_idx].mean(axis=0)
    pixel_means_malignant = X_train[malignant_idx].mean(axis=0)
    axes[1, 0].hist(pixel_means_benign, bins=50, alpha=0.6, label='Benign', color='#2ecc71', density=True)
    axes[1, 0].hist(pixel_means_malignant, bins=50, alpha=0.6, label='Malignant', color='#e74c3c', density=True)
    axes[1, 0].set_title("Distribución de Intensidad Promedio", fontweight='bold')
    axes[1, 0].set_xlabel("Intensidad de Píxel")
    axes[1, 0].legend()

    mean_img_benign = X_train[benign_idx].mean(axis=0).reshape(IMG_SIZE, IMG_SIZE, 3).astype(np.uint8)
    mean_img_malignant = X_train[malignant_idx].mean(axis=0).reshape(IMG_SIZE, IMG_SIZE, 3).astype(np.uint8)
    axes[1, 1].imshow(mean_img_benign)
    axes[1, 1].set_title("Imagen Promedio Benign", fontweight='bold')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(mean_img_malignant)
    axes[1, 2].set_title("Imagen Promedio Malignant", fontweight='bold')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "plots", "eda_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Gráficos EDA guardados en plots/eda_analysis.png")


def train_model(X_train, y_train):
    print("\n" + "=" * 60)
    print("FASE III: ENTRENAMIENTO DEL MODELO")
    print("=" * 60)

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print("Entrenando Regresión Logística (max_iter=1000, solver='lbfgs')...")
    model = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        C=1.0,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train_encoded)

    print("Modelo entrenado exitosamente.")
    print(f"  Coeficientes shape: {model.coef_.shape}")
    print(f"  Intercept: {model.intercept_[0]:.4f}")

    return model, scaler, le


def visualize_coefficients(model):
    print("\nGenerando mapa de coeficientes...")
    coefs = model.coef_.reshape(IMG_SIZE, IMG_SIZE, 3)
    coefs_gray = np.mean(np.abs(coefs), axis=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Análisis de Coeficientes del Modelo", fontsize=14, fontweight='bold')

    im1 = axes[0].imshow(coefs_gray, cmap='RdBu_r', aspect='auto')
    axes[0].set_title("Peso Absoluto por Píxel (Promedio RGB)", fontweight='bold')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    im2 = axes[1].imshow(coefs.mean(axis=2), cmap='RdBu_r', aspect='auto', vmin=-coefs_gray.max(), vmax=coefs_gray.max())
    axes[1].set_title("Peso Promedio por Píxel", fontweight='bold')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "plots", "coefficients.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Mapa de coeficientes guardado en plots/coefficients.png")


def evaluate_model(model, scaler, le, X_test, y_test):
    print("\n" + "=" * 60)
    print("FASE IV: EVALUACIÓN DEL MODELO")
    print("=" * 60)

    y_test_encoded = le.transform(y_test)
    X_test_scaled = scaler.transform(X_test)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test_encoded, y_pred)
    prec = precision_score(y_test_encoded, y_pred)
    rec = recall_score(y_test_encoded, y_pred)
    f1 = f1_score(y_test_encoded, y_pred)
    auc = roc_auc_score(y_test_encoded, y_prob)

    print("\n--- Métricas de Evaluación ---")
    print(f"  Exactitud (Accuracy):  {acc:.4f}")
    print(f"  Precisión (Precision): {prec:.4f}")
    print(f"  Sensibilidad (Recall): {rec:.4f}")
    print(f"  F1-Score:              {f1:.4f}")
    print(f"  AUC-ROC:               {auc:.4f}")

    print("\n--- Classification Report ---")
    print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test_encoded, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Evaluación del Modelo - Regresión Logística", fontsize=14, fontweight='bold')

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[0])
    axes[0].set_title("Matriz de Confusión", fontweight='bold')
    axes[0].set_ylabel("Real")
    axes[0].set_xlabel("Predicho")

    fpr, tpr, _ = roc_curve(y_test_encoded, y_prob)
    axes[1].plot(fpr, tpr, color='#e74c3c', linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
    axes[1].plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
    axes[1].set_title("Curva ROC", fontweight='bold')
    axes[1].set_xlabel("Tasa de Falsos Positivos (FPR)")
    axes[1].set_ylabel("Tasa de Verdaderos Positivos (TPR)")
    axes[1].legend(loc='lower right', fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "plots", "evaluation.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("\nGráficos de evaluación guardados en plots/evaluation.png")

    metrics = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'AUC-ROC': auc
    }
    return metrics


def save_model(model, scaler, le):
    print("\n" + "=" * 60)
    print("GUARDANDO MODELO")
    print("=" * 60)
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, os.path.join(MODEL_DIR, "melanoma_logistic_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))

    joblib.dump({
        'img_size': IMG_SIZE,
        'model_file': "melanoma_logistic_model.pkl",
        'scaler_file': "scaler.pkl",
        'label_encoder_file': "label_encoder.pkl"
    }, os.path.join(MODEL_DIR, "model_config.pkl"))

    print("Modelo guardado en model/melanoma_logistic_model.pkl")
    print("Scaler guardado en model/scaler.pkl")
    print("Label Encoder guardado en model/label_encoder.pkl")
    print("Config guardado en model/model_config.pkl")


def main():
    print("=" * 60)
    print("  CLASIFICACION DE MELANOMA - REGRESION LOGISTICA")
    print("  Benign vs Malignant")
    print("=" * 60)
    print()

    X_train, y_train, X_test, y_test = load_dataset()

    perform_eda(X_train, y_train)

    model, scaler, le = train_model(X_train, y_train)

    visualize_coefficients(model)

    metrics = evaluate_model(model, scaler, le, X_test, y_test)

    save_model(model, scaler, le)

    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)
    print(f"  Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall:    {metrics['Recall']:.4f}")
    print(f"  F1-Score:  {metrics['F1-Score']:.4f}")
    print(f"  AUC-ROC:   {metrics['AUC-ROC']:.4f}")
    print("\n¡Entrenamiento completado exitosamente!")


if __name__ == "__main__":
    main()
