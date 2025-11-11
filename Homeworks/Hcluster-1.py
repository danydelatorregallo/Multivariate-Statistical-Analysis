#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Rocío Carrasco 
"""

#%% Importar librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering
#Configuración de las gráficas
plt.rcParams['figure.figsize'] = (13, 13)
plt.style.use('ggplot')

#%% Descripción de la base de datos
data=pd.read_csv('analisis.csv')
#Rasgos de personalidad de usuarios de Twitter
#140 famosos del mundo de diferentes áreas: deporte, cantantes, actores, etc.
#Basado en una metodología de psicología conocida como “Ocean: The Big Five” 
#tenemos como características de entrada:
    
#usuario (el nombre en Twitter)
#op = Openness to experience – grado de aperura a la experiencia (intuitivo/curioso vs conciente/cauteloso)
#co =Conscientiousness – grado de responsabilidad (eficiente/organizado vs extravagante/descuidado)
#ex = Extraversion – grado de extraversión(sociable/enérgico vs solitario/reservado)
#ag = Agreeableness – grado de amabilidad (amigable/compasivo vs desafiante/insensible)
#ne = Neuroticism, – grado de neuroticismo/estabilidad emocional (susceptible/nervioso vs resistente/seguro)
#Wordcount – Cantidad promedio de palabras usadas en sus tweets
#Categoria – Actividad laboral del usuario (1:actor, 2:cantante, 3:Modelo,4:Tv/series
#5:Radio, 6:Tecnología, 7:Deportes,8:Política y 9:Escritor)

#Resumen estadístico
res=data.describe()
#Número de individuos por categorías
print(data.groupby('categoria').size())

#%%Selección de variables
X = np.array(data[["op","ex","ag"]])
Y = np.array(data['categoria'])


#%% Gráfica de variables con categorías

# Colores asignados a cada categoría
colores = ['blue', 'red', 'green', 'cyan', 'yellow', 'orange', 'black', 'pink', 'brown', 'purple']
categorias = data['categoria'].unique()

# Creando la figura y el eje 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Graficando los puntos y añadiendo etiquetas para la leyenda
for categoria, color in zip(categorias, colores):
    # Filtrando los datos por categoría
    datos_categoria = data[data['categoria'] == categoria]
    
    # Extrayendo las coordenadas X, Y, Z
    x = datos_categoria['op']  
    y = datos_categoria['ex']  
    z = datos_categoria['ag']  

    # Graficando los puntos de cada categoría
    ax.scatter(x, y, z, color=color, label=f"Categoria {categoria}")
ax.legend()
ax.set_xlabel('Oppeness')
ax.set_ylabel('Extraversion')
ax.set_zlabel('Agreeableness')
plt.show()


#%% Clusterig jerarquico scipy
HC= hierarchy.linkage(X,metric='euclidean',method='complete')
#method:single,complete,avergae,centroid, ward,mean
#metric: minkowski, cityblock,euclidean,cosine,correlación,
#jaccard,chebyshev,hamming,mahalanobis,etc...
#note:centroid, median y ward sólo con Euclidean

#Dendograma
plt.title('Dendrograma Completo')
plt.xlabel('Índice de la muestra')
plt.ylabel('Distancia')
dn = hierarchy.dendrogram(HC)
plt.show()

#%% Modificar el aspecto del dendrograma
#plt.figure(figsize=(25,15))
plt.figure()
plt.title('Dendrograma Truncado')
plt.xlabel('índice de la muestra')
plt.ylabel('Distancia')
dn = hierarchy.dendrogram(HC, truncate_mode='level',p=3)
plt.show()

#%% Clustering jerárquico sklearn
cluster = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='complete')  
grupos=cluster.fit_predict(X)

#%% Gráfica con los 3 clústers

colores = ['red', 'orange', 'blue']
asignar = [colores[grupo] for grupo in grupos]

# Creando la gráfica 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=asignar, s=50)

for i, color in enumerate(colores):
    ax.scatter([], [], [], color=color, label=f'Grupo {i+1}')

# Añadiendo la leyenda y etiquetas
ax.set_xlabel('Oppeness')
ax.set_ylabel('Extraversion')
ax.set_zlabel('Agreeableness')
ax.legend()

plt.show()


#%% Análisis de los clusters
def analizar_cluster(X, data, cluster_id):
    """Analiza un cluster específico."""
    idx = grupos == cluster_id
    subdata = pd.DataFrame(X[idx])

    # Estadísticas descriptivas
    res = subdata.describe()

    # Conteos de categorías y usuarios
    categoria_counts = pd.value_counts(data['categoria'][idx])
    usuario_counts = pd.value_counts(data['usuario'][idx])

    return res, categoria_counts, usuario_counts

# Asumiendo que 'grupos' es una lista o array con los identificadores de cluster
numero_de_clusters = 3

for i in range(numero_de_clusters):
    res, categoria_counts, usuario_counts = analizar_cluster(X, data, i)

    print(f"Cluster {i+1} - Análisis")
    print("--------------------------")
    print("Estadísticas Descriptivas:")
    print(res)
    print("\nConteo de Categorías:")
    print(categoria_counts)
    print("\nConteo de Usuarios:")
    print(usuario_counts)
    print("\n")
