import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

# Lectura de datos

df = pd.read_csv("Datos/ethylene_CO.txt", delim_whitespace=True, header=None) #Lee el txt y lo convierte a cvs
# Cambia el nombre de las columnas
columnas = ["Tiempo (s)", "CO (ppm)", "Etileno (ppm)"] + [f"Sensor {i+1}" for i in range(16)]
df.columns = columnas
#Elimar las columnas indicadas por la profa :v
df = df.drop(columns=["Tiempo (s)", "CO (ppm)", "Etileno (ppm)"])
#No es necesario solo lo convierte a cvs de manera externa
#df.to_csv("archivo_convertido.csv", index=False)
print(df.head()) #Imprime las primeras 5 filas del dataframe para verificar que se haya leído correctamente

# Algoritmo de agrupamiento jerárquico
df_sample = df.sample(n=10500) # Toma una muestra aleatoria de 10500 filas del dataframe original
Z = linkage(df_sample, method='ward') # Realiza el agrupamiento jerárquico utilizando el método de Ward
np.set_printoptions(threshold=np.inf)  # Esto hará que se imprima toda la matriz
print(Z)
# Dendrograma completo
plt.figure(figsize=(10, 7)) #Tamaño de la interfaz
dendrogram(Z)
plt.title("Dendrograma de agrupamiento jerárquico completo")
plt.xlabel("Índice de muestra")
plt.ylabel("Distancia o disimilitud")
plt.show()

#Dendrograma recortado
clusters = fcluster(Z, t=3, criterion='maxclust')
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title("Dendrograma de agrupamiento jerárquico")
plt.xlabel("Índice de muestra")
plt.ylabel("Distancia o disimilitud")
plt.show()


"""
Nomneclatura de los datos:
[9.65900000e+03 1.04160000e+04 1.34046037e+01 2.00000000e+00]
9.65900000e+03 = El primer índice de los registros que se están combinando
1.04160000e+04 = El segundo índice de los registros que se están combinando
1.34046037e+01 = La distancia entre los registros que se están combinando
2.00000000e+00 = El número de elementos en el nuevo cluster combinado
"""
