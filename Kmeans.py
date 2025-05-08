import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, accuracy_score
from scipy.spatial.distance import cdist
from sklearn.naive_bayes import GaussianNB
from collections import Counter
from sklearn.model_selection import train_test_split

# === Funciones auxiliares ===

def dunn_index(X, labels):
    unique_labels = np.unique(labels)
    clusters = [X[labels == lbl] for lbl in unique_labels]
    intra_dists = [np.max(cdist(c, c)) for c in clusters if len(c) > 1]
    max_intra = np.max(intra_dists) if intra_dists else 0
    inter_dists = [np.min(cdist(clusters[i], clusters[j]))
                   for i in range(len(clusters)) for j in range(i + 1, len(clusters))]
    min_inter = np.min(inter_dists) if inter_dists else 0
    return np.inf if max_intra == 0 else min_inter / max_intra

def hardlim(x):
    return np.where(x >= 0, 1, 0)

class HardlimNN:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(output_dim, input_dim)
        self.b = np.random.randn(output_dim)

    def predict(self, X):
        return hardlim(np.dot(X, self.W.T) + self.b)

    def fit(self, X, y, epochs=100, lr=0.01):
        y_bin = np.zeros((len(y), np.max(y) + 1))
        y_bin[np.arange(len(y)), y] = 1  # One-hot encoding

        for epoch in range(epochs):
            for i in range(len(X)):
                xi = X[i]
                target = y_bin[i]
                output = hardlim(np.dot(self.W, xi) + self.b)
                error = target - output
                self.W += lr * error[:, np.newaxis] * xi
                self.b += lr * error

# === Paso 1: Cargar datos ===

df = pd.read_csv("Datos/datos_combinados_formato_final.txt", delim_whitespace=True, header=None)
columnas = ["Tiempo (s)", "CO (ppm)", "Etileno (ppm)"] + [f"Sensor {i+1}" for i in range(16)]
df.columns = columnas
X = df.drop(columns=["Tiempo (s)", "CO (ppm)", "Etileno (ppm)"]).to_numpy()

# === Paso 2: Evaluar KMeans con métricas ===

range_k = [2, 3, 4]
for k in range_k:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)
    dunn = dunn_index(X, labels)
    conteo = dict(Counter(labels))

    print(f"\n=== k = {k} ===")
    print(f"Calinski-Harabasz: {ch:.2f}")
    print(f"Davies-Bouldin:    {db:.2f}")
    print(f"Dunn:              {dunn:.4f}")
    print("N° de puntos por cluster:")
    for cl, cnt in conteo.items():
        print(f"  • Cluster {cl}: {cnt}")

# === Paso 3: Aplicar Bayes y RNA usando el 20% de datos etiquetados por KMeans ===

for k in range_k:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    full_labels = km.fit_predict(X)

    # Dividir 20% para entrenamiento (con etiquetas de KMeans), 80% para prueba
    X_train, X_test, y_train, y_test = train_test_split(X, full_labels, test_size=0.8, random_state=42, stratify=full_labels)

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    acc_nb = accuracy_score(y_test, y_pred_nb)

    # Red neuronal con hardlim
    nn = HardlimNN(input_dim=X.shape[1], output_dim=k)
    nn.fit(X_train, y_train, epochs=100, lr=0.01)
    y_pred_nn = np.argmax(nn.predict(X_test), axis=1)
    acc_nn = accuracy_score(y_test, y_pred_nn)

    print(f"\n-- Clasificación usando solo el 20% de datos etiquetados por KMeans (k={k}) --")
    print(f"Precisión Naive Bayes:        {acc_nb:.2f}")
    print(f"Precisión Red Neuronal (hardlim): {acc_nn:.2f}")
