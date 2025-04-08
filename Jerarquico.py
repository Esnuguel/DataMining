import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances


df = pd.read_csv("Datos/ethylene_CO.txt", delim_whitespace=True,header=None)
columnas = ["Tiempo (s)", "CO (ppm)", "Etileno (ppm)"] + [f"Sensor {i+1}" for i in range(16)]
df.columns = columnas
df = df.drop(columns=["Tiempo (s)", "CO (ppm)", "Etileno (ppm)"])
print(df.head())
#print(df.shape)

df_sample = df.head(10500) #Toa las primeras 10,500 filas
#df_sample = df.sample(n=10500) #Toma una muestra aleatoria de 10,500 filas del dataframe original

# Normalizar los datos
normalized_data = normalize(df_sample) # Normaliza los datos a lo largo de las columnas
df_normalized = pd.DataFrame(normalized_data, columns=[f"Sensor {i+1}" for i in range(16)]) # Crea un nuevo DataFrame con los datos normalizados
print(df_normalized.head())

#Calculo de las distancias entre puntos
distances = pairwise_distances(normalized_data, metric='euclidean')
distance_df = pd.DataFrame(distances)
print("Matriz de distancias euclidianas entre los puntos:")
print(distance_df)

#Creacion del dendrograma
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(df_normalized, method='ward'))
plt.axhline(y=30, color='r', linestyle='--')
plt.axhline(y=20, color='g', linestyle='--')
plt.axhline(y=12, color='b', linestyle='--')
plt.show()

#Agrupamiento de clusteres
np.set_printoptions(threshold=np.inf)  # Mostrar todo el array sin recortes
cluster1 = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')
cluster2 = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
cluster3 = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')
labels1 = cluster1.fit_predict(df_normalized)
labels2 = cluster2.fit_predict(df_normalized)
labels3 = cluster3.fit_predict(df_normalized)
print("Partición con 4 grupos \n",labels1)
print("Partición con 3 grupos \n",labels2)
print("Partición con 2 grupos \n",labels3) 