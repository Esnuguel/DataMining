import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Lectura de datos
df = pd.read_csv("./Data/ethylene_CO.txt", delim_whitespace=True, header=None)

# Renombrar columnas
columnas = ["Tiempo (s)", "CO (ppm)", "Etileno (ppm)"] + [f"Sensor {i+1}" for i in range(16)]
df.columns = columnas

# Eliminar columnas no utilizadas
df = df.drop(columns=["Tiempo (s)", "CO (ppm)", "Etileno (ppm)"])

# Usar solo el 10% de los datos para agilizar el cálculo
df_muestra, _ = train_test_split(df, test_size=0.9, random_state=42)

print(f"Tamaño del conjunto original: {len(df)}")
print(f"Tamaño del conjunto de muestra (10%): {len(df_muestra)}")

# Método del codo para determinar el número óptimo de clusters
inercias = []
k_range = range(2, 10)  # Probar de 1 a 15 clusters

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_muestra)
    inercias.append(kmeans.inertia_)

# Graficar el método del codo
plt.figure(figsize=(10, 6))
plt.plot(k_range, inercias, 'bo-')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Inercia (Suma de distancias al cuadrado)')
plt.title('Método del Codo para determinar k óptimo')
plt.grid(True)
plt.show()
