import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import cdist
from collections import Counter

# Función para calcular el índice de Dunn
def dunn_index(X, labels):
    unique_cluster_labels = np.unique(labels)
    clusters = [X[labels == label] for label in unique_cluster_labels]
    
    # Intra-cluster distances (máxima distancia dentro de cada cluster)
    intra_dists = [np.max(cdist(cluster, cluster)) for cluster in clusters if len(cluster) > 1]
    max_intra_dist = np.max(intra_dists) if intra_dists else 0

    # Inter-cluster distances (mínima distancia entre distintos clusters)
    inter_dists = []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            dist = np.min(cdist(clusters[i], clusters[j]))
            inter_dists.append(dist)
    min_inter_dist = np.min(inter_dists) if inter_dists else 0

    if max_intra_dist == 0:
        return np.inf
    
    return min_inter_dist / max_intra_dist

# Cargar y preparar datos
df = pd.read_csv("Datos/datos_combinados_formato_final.txt", delim_whitespace=True, header=None)
columnas = ["Tiempo (s)", "CO (ppm)", "Etileno (ppm)"] + [f"Sensor {i+1}" for i in range(16)]
df.columns = columnas
df = df.drop(columns=["Tiempo (s)", "CO (ppm)", "Etileno (ppm)"])

# Dendrograma
plt.figure(figsize=(10, 7))  
plt.title("Dendrograma")  
dend = shc.dendrogram(shc.linkage(df, method='ward'))
plt.show()

# Aplicar clustering y calcular índices de validez
for n_clusters in [2, 3, 4]:
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    cluster_labels = clustering.fit_predict(df)

    # Calcular los índices
    ch_score = calinski_harabasz_score(df, cluster_labels)
    db_score = davies_bouldin_score(df, cluster_labels)
    dunn = dunn_index(df.to_numpy(), np.array(cluster_labels))
    
    # Contar cuántos datos hay en cada grupo
    conteo = Counter(cluster_labels)
    print(f"\nNúmero de datos en cada grupo para {n_clusters} clusters:")
    for grupo, cantidad in conteo.items():
        print(f"Grupo {grupo}: {cantidad} datos")
    
    # Mostrar los índices
    print(f"Índice de Calinski-Harabasz: {ch_score:.2f}")
    print(f"Índice de Davies-Bouldin: {db_score:.2f}")
    print(f"Índice de Dunn: {dunn:.4f}")