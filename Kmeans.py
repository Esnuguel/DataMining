import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import cdist
from collections import Counter

# Función para calcular el índice de Dunn
def dunn_index(X, labels):
    unique_labels = np.unique(labels)
    clusters = [X[labels == lbl] for lbl in unique_labels]
    
    # Intra-cluster máximo
    intra_dists = [np.max(cdist(c, c)) for c in clusters if len(c) > 1]
    max_intra = np.max(intra_dists) if intra_dists else 0

    # Inter-cluster mínimo
    inter_dists = []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            inter_dists.append(np.min(cdist(clusters[i], clusters[j])))
    min_inter = np.min(inter_dists) if inter_dists else 0

    if max_intra == 0:
        return np.inf
    return min_inter / max_intra

# 1) Cargar y preparar datos
df = pd.read_csv("Datos/datos_combinados_formato_final.txt",delim_whitespace=True, header=None)
columnas = ["Tiempo (s)", "CO (ppm)", "Etileno (ppm)"] + [f"Sensor {i+1}" for i in range(16)]
df.columns = columnas

# Quitamos tiempo y gases, dejando solo sensores
X = df.drop(columns=["Tiempo (s)", "CO (ppm)", "Etileno (ppm)"]).to_numpy()

# 2) Explorar varios k en una sola pasada
range_k = [2, 3, 4]   # ajusta aquí tus valores de k
resultados = []

for k in range_k:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    
    # Índices de validación
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)
    dunn = dunn_index(X, labels)
    
    # Conteo de objetos por cluster
    conteo = dict(Counter(labels))
    
    # Guardamos resultados
    resultados.append({
        'k': k,
        'Calinski-Harabasz': ch,
        'Davies-Bouldin': db,
        'Dunn': dunn,
        'Cluster counts': conteo
    })

# 3) Mostrar resultados
for res in resultados:
    print(f"\n=== k = {res['k']} ===")
    print(f"Calinski-Harabasz: {res['Calinski-Harabasz']:.2f}")
    print(f"Davies-Bouldin:    {res['Davies-Bouldin']:.2f}")
    print(f"Dunn:              {res['Dunn']:.4f}")
    print("N° de puntos por cluster:")
    for cl, cnt in res['Cluster counts'].items():
        print(f"  • Cluster {cl}: {cnt}")
