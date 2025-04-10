import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from collections import Counter


df = pd.read_csv("Datos/datos_combinados_formato_final.txt", delim_whitespace=True,header=None)
columnas = ["Tiempo (s)", "CO (ppm)", "Etileno (ppm)"] + [f"Sensor {i+1}" for i in range(16)]
df.columns = columnas
df = df.drop(columns=["Tiempo (s)", "CO (ppm)", "Etileno (ppm)"])
print(df.head())
#print(df.shape)

#df_sample = df.head(50000) #Toa las primeras 10,500 filas
#df_sample = df.sample(n=10500) #Toma una muestra aleatoria de 10,500 filas del dataframe original

# Normalizar los datos
#normalized_data = normalize(df_sample) # Normaliza los datos a lo largo de las columnas
#f_normalized = pd.DataFrame(normalized_data, columns=[f"Sensor {i+1}" for i in range(16)]) # Crea un nuevo DataFrame con los datos normalizados
#print(df_normalized.head())
"""
#Calculo de las distancias entre puntos
distances = pairwise_distances(df, metric='euclidean')
distance_df = pd.DataFrame(distances)
print("Matriz de distancias euclidianas entre los puntos:")
print(distance_df)
"""
#Creacion del dendrograma
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(df, method='ward'))
#plt.axhline(y=30, color='r', linestyle='--')
#plt.axhline(y=20, color='g', linestyle='--')
#plt.axhline(y=12, color='b', linestyle='--')
#plt.show()

#Agrupamiento de clusteres
"""
np.set_printoptions(threshold=np.inf)  # Mostrar todo el array sin recortes
cluster1 = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')
cluster2 = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
cluster3 = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')
labels1 = cluster1.fit_predict(df_sample)
labels2 = cluster2.fit_predict(df_sample)
labels3 = cluster3.fit_predict(df_sample)
print("Partición con 4 grupos \n",labels1)
print("Partición con 3 grupos \n",labels2)
print("Partición con 2 grupos \n",labels3) 
"""

# Aplicar AgglomerativeClustering para 2, 3 y 4 clusters
for n_clusters in [2, 3, 4]:
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')  # ¡sin affinity ni metric!
    cluster_labels = clustering.fit_predict(df)
    
    # Contar cuántos datos hay en cada grupo
    conteo = Counter(cluster_labels)
    print(f"\nNúmero de datos en cada grupo para {n_clusters} clusters:")
    for grupo, cantidad in conteo.items():
        print(f"Grupo {grupo}: {cantidad} datos")