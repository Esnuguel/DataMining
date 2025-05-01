import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# Lectura de datos
df = pd.read_csv("./Data/ethylene_CO.txt", delim_whitespace=True, header=None)

# Renombrar columnas
columnas = ["Tiempo (s)", "CO (ppm)", "Etileno (ppm)"] + [f"Sensor {i+1}" for i in range(16)]
df.columns = columnas

# Eliminar columnas innecesarias
df = df.drop(columns=["Tiempo (s)", "CO (ppm)", "Etileno (ppm)"])

# Usar solo el 10% de los datos para agilizar el proceso
df_muestra, _ = train_test_split(df, test_size=0.9, random_state=42)

# Aplicar KMeans con k definido (ajusta este valor según los resultados del método del codo)
k = 4  
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(df_muestra)

# Obtener etiquetas y centroides
df_muestra['Cluster'] = kmeans.labels_
centroids = kmeans.cluster_centers_

print("Centroides:")
print(centroids)

# Mostrar información de cada cluster y guardar en CSV
for i in range(k):
    cluster_members = df_muestra[df_muestra['Cluster'] == i]
    print(f"\n--- Cluster {i} ---")
    print(f"Número de objetos: {len(cluster_members)}")
    print("Centroide:")
    print(centroids[i])
    print("Objetos del cluster:")
    print(cluster_members.drop('Cluster', axis=1))
