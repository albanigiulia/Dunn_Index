import numpy as np
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import torch
from torchmetrics.clustering import DunnIndex
###############################

# calcolo indice Dunn
dunn_index = DunnIndex(p=2)
###############################

# calcolo labels kmeans
def label(matrix):
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
    kmeans.fit(matrix)
    return kmeans.labels_
###############################

#valori 
colonne = 5 
decimali = 2 
lower_limit1 = 9990
upper_limit1 = 10000
lower_limit2 = 1 
upper_limit2 = 10
n = 100


#matrice due cluster separati
def create_list(n):
    if n <= 0:
        raise ValueError("n deve essere maggiore di 0")

    final_list = []
    half_n = n // 2
    for _ in range(half_n):
        row = [round(random.uniform(lower_limit1, upper_limit1), decimali) for _ in range(colonne)]
        final_list.append(row)
    for _ in range(n - half_n):
        row = [round(random.uniform(lower_limit2, upper_limit2), decimali) for _ in range(colonne)]
        final_list.append(row)
    return final_list
ordinata = create_list(n)

# grafico
x = [item[0] for item in ordinata]
y = [item[1] for item in ordinata]
plt.scatter(x, y, s=10, color='black', marker='o')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Grafico di dispersione a due dimensioni')
plt.grid()
plt.show()

#dunn
print("Dunn index - matrice cluster separati: ")
labels_ordinata= label(ordinata)
M2 = torch.tensor(ordinata)
labels = torch.tensor(labels_ordinata)
result = dunn_index(M2, labels).item()
print(result)
###############################

#valori 
colonne = 5
lower_limit = 0
upper_limit = 1000
n = 100

#matrice valori sparsi
def crea_matrice(n):
    # Creiamo una lista di n righe e 5 colonne con valori casuali da 0 a 10
    matrice = [[random.uniform(lower_limit, upper_limit) for _ in range(colonne)] for _ in range(n)]
    return matrice
matrice_random = crea_matrice(n)
# Creazione della lista degli indici (x)
x = range(n)  # Indici per le righe

# grafico
x = [item[0] for item in matrice_random]
y = [item[1] for item in matrice_random]
plt.scatter(x, y, s=10, color='black', marker='o')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Grafico di dispersione a due dimensioni')
plt.grid()
plt.show()

#dunn
print("Dunn index - matrice valori sparsi: ")
labels_random = label(matrice_random)
M2 = torch.tensor(matrice_random)
labels = torch.tensor(labels_random)
result = dunn_index(M2, labels).item()
print(result)