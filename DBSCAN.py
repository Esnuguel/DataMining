import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN


ruta = './Datos/ethylene_CO.txt'
df = pd.read_csv(ruta, sep="\s+", skiprows=1, engine="python")  
df = df.iloc[:, 3:]
df_sample = df.sample(frac=0.1, random_state=1)

#Escalar y dividir
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_sample)
particiones = np.array_split(X_scaled, 3)
particion = particiones[2]  # Elegir la tercera partición

#Calcular epsilon
min_samples = 10
nn = NearestNeighbors(n_neighbors=min_samples, algorithm='kd_tree')  
nn.fit(particion)
distances, _ = nn.kneighbors(particion)
k_distances = np.sort(distances[:, min_samples - 1])
epsilon = k_distances[int(len(k_distances) * 0.1)] 

#DBSCAN
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
dbscan.fit(particion)
labels = dbscan.labels_

#Métricas
n_ruidos = np.sum(labels == -1)
n_grupos = len(set(labels)) - (1 if -1 in labels else 0)
distribucion = np.bincount(labels[labels >= 0])

# Clasificación detallada
ruidos = [f"O{i}" for i, lbl in enumerate(labels) if lbl == -1]
grupos = {f"G{g}": [] for g in range(n_grupos)}
for i, lbl in enumerate(labels):
    if lbl >= 0:
        grupos[f"G{lbl}"].append(f"O{i}")

# Puntos centrales y bordes
n_bordes = np.sum((labels >= 0) & (np.bincount(labels + 1)[labels + 1] > min_samples))
n_centrales = np.sum(labels >= 0) - n_bordes

# Guardar la salida en un archivo
with open("salida.txt", "a", encoding="utf-8") as f:
    f.write("\n" + "="*72 + "\n")
    f.write("RESUMEN DE AGRUPAMIENTO DBSCAN - PARTICIÓN 3".center(72) + "\n")
    f.write("="*72 + "\n")
    f.write(f"Épsilon (ε): {epsilon:.4f}\n")
    f.write(f"Mínimas muestras (min_samples): {min_samples}\n")
    f.write(f"Número de grupos encontrados (G): {n_grupos}\n")
    f.write(f"Numero de bordes encontrados (B): {n_bordes}\n")
    f.write("-"*72 + "\n")
    f.write(f"R = {n_ruidos}, PC = {n_centrales}, B = {n_bordes}\n\n\n\n\n\n")  # 5 saltos de línea
    f.write("="*72 + "\n\n")

    f.write("AGRUPACIÓN DETALLADA:\n")
    f.write(f"R (Ruido) = {{{', '.join(ruidos)}}}\n\n\n\n\n")  # Objetos en ruido
    f.write(f"PC (Puntos Centrales) = {{{', '.join([f'O{i}' for i in range(len(labels)) if labels[i] >= 0 and np.bincount(labels + 1)[labels[i] + 1] > min_samples])}}}\n\n\n\n\n")  # Puntos centrales
    f.write(f"B (Puntos de Borde) = {{{', '.join([f'O{i}' for i in range(len(labels)) if labels[i] >= 0 and np.bincount(labels + 1)[labels[i] + 1] <= min_samples])}}}\n\n\n\n\n")  # Puntos de borde
    for g, objetos in grupos.items():
        f.write(f"{g} = {{{', '.join(objetos)}}}\n")

    f.write("\n" + "-"*72 + "\n")
    f.write("ASIGNACIÓN DE GRUPOS PARA TODOS LOS OBJETOS:\n")
    f.write("-"*72 + "\n")
    
    
    for i, label in enumerate(labels):
        grupo = "R" if label == -1 else f"G{label}"
        f.write(f"O{i+1} -> {grupo}\n")

print("Salida guardada en 'salida.txt'")
print("Fin del programa.")
