# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 12:19:45 2025

@author: LUISALVARADO
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#%% Matriz original de ejemplo (3 observaciones, 2 variables)
X = np.array([
    [3, 0],
    [1, -1],
    [-2, 2]
])
#%% Aplicar PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
#%% Mostrar resultados
print("Matriz original:")
print(X)
print("\nMatriz de covarianza:\n")
print(pca.get_covariance())
print("\nEigenvalores:")
print(pca.explained_variance_)
print("\nEigenvectores:")
print(pca.components_)
print("\nComponentes principales (transformación):")
print(X_pca)
#%% Varianza explicada por cada componente principal
eigvals, eigvecs = np.linalg.eig(pca.get_covariance()) # Get vectors for covariance matrix
eigvals=np.abs(eigvals)
indx = np.argsort(eigvals)[::-1] # sort
porcentaje = eigvals[indx]/np.sum(eigvals) # convert to %
porcent_acum = np.cumsum(porcentaje) # Accumulate
print(porcent_acum)
#%% Graficar
X_centered = X - X.mean(axis=0)
X_reconstructed = pca.inverse_transform(np.column_stack((X_pca[:,0], np.zeros_like(X_pca[:,1]))))

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel 1: Datos originales centrados con eigenvectores
ax = axes[0, 0]
ax.scatter(X_centered[:, 0], X_centered[:, 1], color='blue', edgecolor='k')
# Calculamos el centroide de X_centered
origin = np.mean(X_centered, axis=0)
# Dibujamos los eigenvectores (componentes principales) desde el centroide
for i, vector in enumerate(pca.components_):
    ax.arrow(origin[0], origin[1],
             vector[0]*2, vector[1]*2,  # factor de escala para visualización
             color='red', width=0.05, head_width=0.2, length_includes_head=True,
             label=f'CP{i+1}' if i == 0 else None)
ax.set_title("Datos originales centrados con eigenvectores")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.axhline(0, color='gray', linestyle='--')
ax.axvline(0, color='gray', linestyle='--')
ax.grid(True)

# Panel 2: Datos transformados (espacio PCA)
ax = axes[0, 1]
ax.scatter(X_pca[:, 0], X_pca[:, 1], color='blue', edgecolor='k')
ax.set_title("Datos transformados (PCA)")
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.axhline(0, color='gray', linestyle='--')
ax.axvline(0, color='gray', linestyle='--')
ax.grid(True)

# Panel 3: Proyección en PC1 (PC2 descartado)
ax = axes[1, 0]
# Graficamos los valores de PC1 en el eje horizontal y 0 en el vertical
ax.scatter(X_pca[:, 0], np.zeros_like(X_pca[:, 0]), color='blue', edgecolor='k')
for i, (x, _) in enumerate(X_pca):
    ax.text(x, 0.1, f'P{i+1}', fontsize=12)
ax.set_title("Proyección solo en PC1")
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2 = 0")
ax.axhline(0, color='gray', linestyle='--')
ax.axvline(0, color='gray', linestyle='--')
ax.grid(True)

# Panel 4: Back-rotation usando solo PC1 (reconstrucción)
ax = axes[1, 1]
ax.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], color='blue', edgecolor='k', label="Reconstruido (PC1)")
# Opcional: graficar los datos originales centrados para comparación
ax.scatter(X_centered[:, 0], X_centered[:, 1], color='gray', marker='x', label="Original centrado")
ax.set_title("Reconstrucción con solo PC1")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.axhline(0, color='gray', linestyle='--')
ax.axvline(0, color='gray', linestyle='--')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()

