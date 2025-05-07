import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, accuracy_score
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from collections import Counter

# Función para calcular el índice de Dunn
def dunn_index(X, labels):
    unique_labels = np.unique(labels)
    clusters = [X[labels == lbl] for lbl in unique_labels]
    intra_dists = [np.max(cdist(c, c)) for c in clusters if len(c) > 1]
    max_intra = np.max(intra_dists) if intra_dists else 0
    inter_dists = [np.min(cdist(clusters[i], clusters[j]))
                   for i in range(len(clusters)) for j in range(i + 1, len(clusters))]
    min_inter = np.min(inter_dists) if inter_dists else 0
    return np.inf if max_intra == 0 else min_inter / max_intra

# 1) Cargar y preparar datos
df = pd.read_csv("Datos/datos_combinados_formato_final.txt", delim_whitespace=True, header=None)
columnas = ["Tiempo (s)", "CO (ppm)", "Etileno (ppm)"] + [f"Sensor {i+1}" for i in range(16)]
df.columns = columnas
X = df.drop(columns=["Tiempo (s)", "CO (ppm)", "Etileno (ppm)"]).to_numpy()

# 2) KMeans y métricas
range_k = [2, 3, 4]
resultados = []

for k in range_k:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)
    dunn = dunn_index(X, labels)
    conteo = dict(Counter(labels))
    
    resultados.append({
        'k': k,
        'Calinski-Harabasz': ch,
        'Davies-Bouldin': db,
        'Dunn': dunn,
        'Cluster counts': conteo
    })

    print(f"\n=== k = {k} ===")
    print(f"Calinski-Harabasz: {ch:.2f}")
    print(f"Davies-Bouldin:    {db:.2f}")
    print(f"Dunn:              {dunn:.4f}")
    print("N° de puntos por cluster:")
    for cl, cnt in conteo.items():
        print(f"  • Cluster {cl}: {cnt}")

    # 3) Aplicar Bayes y Red Neuronal al 20% del dataset con etiquetas de KMeans
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.20, random_state=42, stratify=labels)

    # Teorema de Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    acc_nb = accuracy_score(y_test, y_pred_nb)

    # Red Neuronal
    mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred_mlp = mlp.predict(X_test)
    acc_mlp = accuracy_score(y_test, y_pred_mlp)

    print(f"\n-- Clasificación sobre 20% de datos usando clusters de KMeans (k={k}) --")
    print(f"Precisión Naive Bayes:  {acc_nb:.2f}")
    print(f"Precisión Red Neuronal: {acc_mlp:.2f}")
