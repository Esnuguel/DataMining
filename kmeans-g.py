import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # Necesario para la gráfica 3D

# ---------------------
# PARTE 1: Lectura de datos, preprocesamiento y clustering
# ---------------------

# Lectura de datos
df = pd.read_csv("./Data/ethylene_CO.txt", delim_whitespace=True, header=None)

# Renombrar columnas: se asignan nombres a Tiempo, CO, Etileno y a 16 sensores
columnas = ["Tiempo (s)", "CO (ppm)", "Etileno (ppm)"] + [f"Sensor {i+1}" for i in range(16)]
df.columns = columnas

# Eliminar columnas innecesarias para el clustering
df = df.drop(columns=["Tiempo (s)", "CO (ppm)", "Etileno (ppm)"])

# Usar solo el 10% de los datos para agilizar el proceso
df_muestra, _ = train_test_split(df, test_size=0.9, random_state=42)

# Aplicar KMeans (ajusta el valor de k según tus resultados o método del codo)
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(df_muestra)

# Obtener etiquetas y centroides
df_muestra['Cluster'] = kmeans.labels_
centroids = kmeans.cluster_centers_

# ---------------------
# PARTE 2: Gráficas de visualización
# ---------------------

# (B) Visualización 3D para los primeros 3 sensores
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Graficar puntos por cluster
for i in range(k):
    cluster_points = df_muestra[df_muestra['Cluster'] == i]
    ax.scatter(cluster_points.iloc[:, 0], 
            cluster_points.iloc[:, 1], 
            cluster_points.iloc[:, 2],
            label=f'Cluster {i}', alpha=0.5)

# Graficar centroides en 3D
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], 
        s=200, c='black', marker='X', label='Centroides')

ax.set_xlabel('Sensor 1')
ax.set_ylabel('Sensor 2')
ax.set_zlabel('Sensor 3')
ax.set_title('Clustering KMeans en 3D')
plt.legend()
plt.tight_layout()
plt.show()
