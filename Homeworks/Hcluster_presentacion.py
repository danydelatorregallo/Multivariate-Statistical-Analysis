# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 22:46:38 2025
@author: Luis Alvarado
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy

#%% 1. Definir los datos y etiquetas
labels = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
X = np.array([
    [0.40, 0.53],  
    [0.22, 0.38],  
    [0.35, 0.32],  
    [0.26, 0.19],  
    [0.08, 0.41],  
    [0.45, 0.30],  
])

#%% 2. Calcular las fusiones (linkage) para single y complete linkage
Z_single   = hierarchy.linkage(X, method='single',   metric='euclidean')
Z_complete = hierarchy.linkage(X, method='complete', metric='euclidean')

#%% 3. Dibujar el dendrograma para single linkage
plt.figure(figsize=(8, 4))
hierarchy.dendrogram(
    Z_single,
    labels=labels,
    orientation='top',
    distance_sort='ascending'
)
plt.title('Dendrograma (Single Linkage)')
plt.xlabel('Puntos')
plt.ylabel('Distancia Euclidiana')
plt.tight_layout()
plt.show()

#%% 4. Dibujar el dendrograma para complete linkage
plt.figure(figsize=(8, 4))
hierarchy.dendrogram(
    Z_complete,
    labels=labels,
    orientation='top',
    distance_sort='ascending'
)
plt.title('Dendrograma (Complete Linkage)')
plt.xlabel('Puntos')
plt.ylabel('Distancia Euclidiana')
plt.tight_layout()
plt.show()

