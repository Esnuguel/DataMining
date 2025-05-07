import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, f1_score, accuracy_score, confusion_matrix

# Suponiendo que df es tu DataFrame y que ya cargaste los datos
df = pd.read_csv("Datos/datos_combinados_formato_final.txt", delim_whitespace=True, header=None)
columnas = ["Tiempo (s)", "CO (ppm)", "Etileno (ppm)"] + [f"Sensor {i+1}" for i in range(16)]
df.columns = columnas

# Quitamos tiempo y gases, dejando solo los sensores
X = df.drop(columns=["Tiempo (s)", "CO (ppm)", "Etileno (ppm)"]).to_numpy()

# Crear las etiquetas reales (alternando entre CO y Metano)
# Suponiendo que tus datos son alternados y que empiezan con 'CO' en la primera fila
y_true = np.array(["CO" if i % 2 == 0 else "Metano" for i in range(len(X))])

# 2. Ejecutar K-means
k = 2  # K-means con 2 clusters
km = KMeans(n_clusters=k, random_state=42, n_init=10)
labels_pred = km.fit_predict(X)  # Predicciones de K-means

# 3. Comparar etiquetas reales con etiquetas predichas
# Convertir las etiquetas predichas de K-means (0, 1) a etiquetas 'CO' y 'Metano'
# Aquí estamos suponiendo que K-means asignó los cluster 0 a CO y el cluster 1 a Metano (esto puede necesitar ajuste)
labels_pred_named = np.array(["CO" if label == 0 else "Metano" for label in labels_pred])

# Evaluar las métricas de clasificación
precision = precision_score(y_true, labels_pred_named, pos_label="CO", average=None)  # Precisión por clase
f1 = f1_score(y_true, labels_pred_named, pos_label="CO", average=None)  # F1 por clase
accuracy = accuracy_score(y_true, labels_pred_named)  # Precisión general
f1_general = f1_score(y_true, labels_pred_named, average="weighted")  # F1 general

# Mostrar los resultados
print(f"Precisión por clase (CO y Metano): {precision}")
print(f"F1-Score por clase (CO y Metano): {f1}")
print(f"Precisión general: {accuracy:.4f}")
print(f"F1-Score general: {f1_general:.4f}")

# Matriz de confusión
cm = confusion_matrix(y_true, labels_pred_named)
print("\nMatriz de Confusión:")
print(cm)
