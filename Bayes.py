import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Cargar los datos
ruta = './Datos/datos_combinados.txt'
datos_combinados = pd.read_csv(ruta, sep="\s+", skiprows=1, engine="python").iloc[:, 3:]

# Ajustamos tamaño
datos_combinados = datos_combinados.iloc[:34999, :]

# Crear etiquetas (0: primeros 17500, 1: siguientes 17499)
etiquetas = np.array([0] * 17500 + [1] * 17499)

# Dividir X y y
X = datos_combinados.values
y = etiquetas

# División entrenamiento/prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo Naive Bayes
modelo = GaussianNB()
modelo.fit(X_train, y_train)

# Obtener probabilidades predichas para cada clase
probs = modelo.predict_proba(X_test)

# Evaluar por cada predicción de MAP
ganados_clase_0 = 0
ganados_clase_1 = 0
correctos_clase_0 = 0
correctos_clase_1 = 0

for i, prob in enumerate(probs):
    prediccion = np.argmax(prob)  # 0 si clase 0, 1 si clase 1
    if prediccion == 0:
        ganados_clase_0 += 1
        if y_test[i] == 0:
            correctos_clase_0 += 1
    else:
        ganados_clase_1 += 1
        if y_test[i] == 1:
            correctos_clase_1 += 1

# Resumen final de las clases ganadas
precision_clase_0 = correctos_clase_0 / ganados_clase_0 if ganados_clase_0 > 0 else 0
precision_clase_1 = correctos_clase_1 / ganados_clase_1 if ganados_clase_1 > 0 else 0

accuracy_total = accuracy_score(y_test, modelo.predict(X_test))

# Determinar cuál clase ganó (la que tiene el mayor accuracy)
if precision_clase_0 > precision_clase_1:
    clase_ganadora = "Clase 0"
    precision_ganadora = precision_clase_0
else:
    clase_ganadora = "Clase 1"
    precision_ganadora = precision_clase_1

# Imprimir resultados
print("\nResumen Final:")
print(f"Total objetos ganados por clase 0: {ganados_clase_0}")
print(f"Total objetos ganados por clase 1: {ganados_clase_1}")
print(f"Precisión de clase 0 usando MAP: {precision_clase_0:.4f}")
print(f"Precisión de clase 1 usando MAP: {precision_clase_1:.4f}")
print(f"\nAccuracy general usando MAP: {accuracy_total:.4f}")
print(f"Clase ganadora según mayor precisión: {clase_ganadora} con una precisión de {precision_ganadora:.4f}")

# Evaluación final del modelo
y_pred = modelo.predict(X_test)
print("\nReporte de Clasificación (MAP):\n", classification_report(y_test, y_pred))
print("\nMatriz de Confusión (MAP):\n", confusion_matrix(y_test, y_pred))

#desviación y media
print("Medias por atributo y por clase:")
print(modelo.theta_)   # medias aprendidas

print("\nDesviaciones estándar por atributo y por clase:")
print(np.sqrt(modelo.var_))  # desviaciones estándar aprendidas




