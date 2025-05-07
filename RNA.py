import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Cargar y preparar datos
df = pd.read_csv("Datos/datos_combinados_formato_final.txt", delim_whitespace=True, header=None)
columnas = ["Tiempo (s)", "CO (ppm)", "Etileno (ppm)"] + [f"Sensor {i+1}" for i in range(16)]
df.columns = columnas
df = df.drop(columns=["Tiempo (s)", "CO (ppm)", "Etileno (ppm)"])
#Converción a numpy
X = df.to_numpy()

#Normalizar entradas, dado que RNA solo trabaja con 0 y 1
X = (X - X.mean(axis=0)) / X.std(axis=0) 
print(X)
"""
X.mean(axis=0) es para calcula la media de cada columna 
axis=0 significa que se va a lo largo de las filas.
por ejemplo:
X = [[1, 10],
     [2, 20],
     [3, 30]]  Daria dar de resultado: [2.0, 20.0]

X.std(axis=0) Realiza lo mismo con la desviación estandar
    Mide qué tanto varían los valores respecto a la media.
    Se hace columna por columna (por sensor).
----------------------------------
    Esta parte genera la nueva desviación estandar
------------------------------------
"""
# ---------------------
# Generar etiquetas reales (intercaladas 0 y 1) 0 pertenece al C0 y 1 al Metano
# ---------------------
n_samples = X.shape[0]
y = np.array([0, 1] * (n_samples // 2))
y = y[:n_samples]  # Asegura que coincida si no es par
print(n_samples)

# ---------------------
# Separar en train/test
# ---------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------
# Implementar perceptrón con hardlim
# ---------------------
def hardlim(x):
    return np.where(x >= 0, 1, 0)


# Inicialización de pesos y sesgo
np.random.seed(0)
n_features = X_train.shape[1]
weights = np.random.randn(n_features)
bias = 0.0
learning_rate = 0.01
epochs = 100

# Entrenamiento
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
print(f"Precisión (accuracy): {acc:.4f}")
print(f"F1-score: {f1:.4f}")