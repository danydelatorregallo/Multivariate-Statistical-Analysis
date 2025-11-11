# -*- coding: utf-8 -*-
"""
@author: Luis Alvarado
"""
import random
import numpy as np
from sklearn.cluster import KMeans
#%%
points = np.array([
    [185, 72],
    [170, 56],
    [168, 60],
    [179, 68],
    [182, 72],
    [188, 77],
], dtype=float)
#%%
K = 2
tol = 1e-3
random.seed(0)
np.random.seed(0)
#%%
def simple_kmeans(points, K, tol=1e-6, max_epochs=10):
    # Inicializo centroides con los primeros K puntos (puedes cambiarlo)
    centroids = points[:K].copy()
    history = [centroids.copy()]

    for epoch in range(max_epochs):
        moved = False
        for x in points:
            # 1) Calcular distancias euclideanas al punto x
            dists = np.linalg.norm(centroids - x, axis=1)
            # 2) Elegir el centroide más cercano
            k = np.argmin(dists)
            # 3) Actualizar ese centroide como (μ_old + x)/2
            new_c = (centroids[k] + x) / 2.0
            if np.linalg.norm(new_c - centroids[k]) > tol:
                moved = True
            centroids[k] = new_c
            history.append(centroids.copy())

        if not moved:
            break

    # asignación final de etiquetas
    dists = np.linalg.norm(points[:,None] - centroids[None,:], axis=2)
    labels = np.argmin(dists, axis=1)
    return centroids, labels, history
#%%
def batch_kmeans(points, K, tol=1e-3, max_iter=100):
    centroids = points[:K].copy()
    history = [centroids.copy()]
    for it in range(max_iter):
        # asignación
        dists  = np.linalg.norm(points[:,None] - centroids[None,:], axis=2)
        labels = np.argmin(dists, axis=1)
        # recomputar centroides
        new_centroids = np.array([
            points[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k]
            for k in range(K)
        ])
        history.append(new_centroids.copy())
        if np.max(np.linalg.norm(new_centroids - centroids, axis=1)) < tol:
            break
        centroids = new_centroids
    return centroids, labels, history
#%%
def online_kmeans(points, K, tol=1e-3, max_epochs=10):
    # Inicializar
    centroids = points[np.random.choice(len(points), K, replace=False)]
    counts = np.zeros(K, dtype=int)
    history = [centroids.copy()]
    for epoch in range(max_epochs):
        moved = False
        for x in points:
            dists = np.linalg.norm(centroids - x, axis=1)
            k = np.argmin(dists)
            counts[k] += 1
            eta = 1.0 / counts[k]
            new_c = centroids[k] + eta * (x - centroids[k])
            if np.linalg.norm(new_c - centroids[k]) > tol:
                moved = True
            centroids[k] = new_c
            history.append(centroids.copy())
        if not moved:
            break
    # Etiquetas finales
    dists = np.linalg.norm(points[:, None] - centroids[None, :], axis=2)
    labels = np.argmin(dists, axis=1)
    return centroids, labels, history
#%% Ejecutar y obtener trazas
c_simple, lab_simple, hist_simple = simple_kmeans(points, K, tol)
c_batch, lab_batch, hist_batch = batch_kmeans(points, K, tol)
c_online, lab_online, hist_online = online_kmeans(points, K, tol)
#%%
print("\n=== Simple k-means trace ===")
for i, cents in enumerate(hist_simple):
    print(f" Step {i:2d}: centroids =\n {cents}")
#%%
print("=== Batch k-means trace ===")
for i, cents in enumerate(hist_batch):
    print(f" Iter {i:2d}: centroids =\n {cents}")
#%%
    print("=== Online k-means trace ===")
    for i, cents in enumerate(hist_online):
        print(f" Iter {i:2d}: centroids =\n {cents}")
#%%
print("\n=== Simple k-means ===")
print("Centroides:\n", c_simple)
print("Etiquetas:", lab_simple)

print("=== Batch k-means ===")
print("Centroides:\n", c_batch)
print("Etiquetas:", lab_batch)

print("=== Online k-means ===")
print("Centroides:\n", c_online)
print("Etiquetas:", lab_online)

#%%
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(points)
kmeans.labels_

