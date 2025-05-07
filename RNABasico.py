import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix

# ---------------------
# Cargar y preparar datos
# ---------------------
df = pd.read_csv("Datos/datos_combinados_formato_final.txt", delim_whitespace=True, header=None)
columnas = ["Tiempo (s)", "CO (ppm)", "Etileno (ppm)"] + [f"Sensor {i+1}" for i in range(16)]
df.columns = columnas
df = df.drop(columns=["Tiempo (s)", "CO (ppm)", "Etileno (ppm)"])
X = df.to_numpy()

# Normalizar entradas
X = (X - X.mean(axis=0)) / X.std(axis=0)
print(X)

# ---------------------
# Generar etiquetas reales
# ---------------------
n_samples = X.shape[0]
y = np.array([0, 1] * (n_samples // 2))
y = y[:n_samples]
print(y)

# ---------------------
# Separar en train/test
# ---------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------
# Perceptrón con hardlim
# ---------------------
def hardlim(x):
    return np.where(x >= 0, 1, 0)

np.random.seed(0)
n_features = X_train.shape[1]
weights = np.random.randn(n_features)
bias = 0.0
learning_rate = 0.01
epochs = 100

for epoch in range(epochs):
    for i in range(X_train.shape[0]):
        xi = X_train[i]
        yi = y_train[i]
        output = hardlim(np.dot(weights, xi) + bias)
        error = yi - output
        weights += learning_rate * error * xi
        bias += learning_rate * error

# ---------------------
# Evaluación
# ---------------------
y_pred = hardlim(np.dot(X_test, weights) + bias)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n🔎 Resultados del perceptrón con hardlim:")
print(f"Accuracy general: {acc:.4f}")
print(f"F1-score general: {f1:.4f}")

# ---------------------
# Métricas por clase
# ---------------------
cm = confusion_matrix(y_test, y_pred)
true_negatives, false_positives, false_negatives, true_positives = cm.ravel()

total_clase_0 = true_negatives + false_negatives
total_clase_1 = true_positives + false_positives

predicciones_correctas_clase_0 = true_negatives
predicciones_correctas_clase_1 = true_positives

precision_clase_0 = precision_score(y_test, y_pred, pos_label=0)
precision_clase_1 = precision_score(y_test, y_pred, pos_label=1)

clase_ganadora = 0 if predicciones_correctas_clase_0 > predicciones_correctas_clase_1 else 1

print("\n📊 Resultados detallados por clase:")
print(f"Total objetos predichos clase 0: {np.sum(y_pred == 0)}")
print(f"Total objetos predichos clase 1: {np.sum(y_pred == 1)}")
print(f"Precisión clase 0: {precision_clase_0:.4f}")
print(f"Precisión clase 1: {precision_clase_1:.4f}")
print(f"Clase ganadora: Clase {clase_ganadora}")