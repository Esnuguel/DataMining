import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Lectura de datos

df = pd.read_csv("Datos/ethylene_CO.txt", delim_whitespace=True, header=None) #Lee el txt y lo convierte a cvs
# Cambia el nombre de las columnas
columnas = ["Tiempo (s)", "CO (ppm)", "Etileno (ppm)"] + [f"Sensor {i+1}" for i in range(16)]
df.columns = columnas
#No es necesario solo lo convierte a cvs de manera externa
#df.to_csv("archivo_convertido.csv", index=False)
print(df.head())

# Algoritmo de agrupamiento jerárquico
df_sample = df.sample(n=10500)
Z = linkage(df_sample, method='ward')
print(Z)