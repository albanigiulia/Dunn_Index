import numpy as np
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import torch
from torchmetrics.clustering import DunnIndex
import time

# Memorizziamo il tempo iniziale
start_time = time.time()
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
lower_limit1 = 100
upper_limit1 = 150 
lower_limit2 = 1 
upper_limit2 = 50 
righe = 100


#matrice due cluster separati
def create_list(n):
    if n <= 0:
        raise ValueError("n deve essere maggiore di 0")

    final_list = []
    half_righe = righe // 2
    for _ in range(half_righe):
        row = [round(random.uniform(lower_limit1, upper_limit1), decimali) for _ in range(colonne)]
        final_list.append(row)
    for _ in range(righe - half_righe):
        row = [round(random.uniform(lower_limit2, upper_limit2), decimali) for _ in range(colonne)]
        final_list.append(row)
    return final_list
ordinata = create_list(righe)

#dunn
print("Dunn index - matrice cluster separati: ")
labels_ordinata= label(ordinata)
M2 = torch.tensor(ordinata)
labels = torch.tensor(labels_ordinata)
result = dunn_index(M2, labels).item()
print(result)

# grafico
variabile1 = True
if variabile1:
    salva_dati1 = True
    x = [item[0] for item in ordinata]
    y = [item[1] for item in ordinata]

    # Imposta i colori per i cluster: Cluster 1 = blu, Cluster 2 = rosso
    colors = ['#1f77b4' if label == 1 else '#ff7f0e' for label in labels_ordinata]

    plt.scatter(x, y, s=10, color=colors, marker='o')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Grafico di dispersione a due dimensioni')
    plt.grid()
    
    if salva_dati1:
        plt.savefig('C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\results\\Immagini\\grafico_separati4.png')
        print("Dati salvati: il grafico è stato salvato come 'grafico_separati1.png'")
    
    plt.show()
###############################

#valori 
colonne = 5
lower_limit = 0
upper_limit = 100
righe = 100 

#matrice valori sparsi
def crea_matrice(righe):
    # Creiamo una lista di n righe e 5 colonne con valori casuali da 0 a 10
    matrice = [[random.uniform(lower_limit, upper_limit) for _ in range(colonne)] for _ in range(righe)]
    return matrice
matrice_random = crea_matrice(righe)
# Creazione della lista degli indici (x)
x = range(righe)  # Indici per le righe

#dunn
print("Dunn index - matrice valori sparsi: ")
labels_random = label(matrice_random)
M2 = torch.tensor(matrice_random)
labels = torch.tensor(labels_random)
result = dunn_index(M2, labels).item()
print(result)

# Grafico
variabile2 = False
if variabile2:
    salva_dati2 = False
    x = [item[0] for item in matrice_random]
    y = [item[1] for item in matrice_random]
    # Imposta i colori per i cluster: Cluster 1 = blu, Cluster 2 = rosso
    colors = ['#1f77b4' if label == 1 else '#ff7f0e' for label in labels]
    plt.scatter(x, y, s=10, color=colors, marker='o')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Grafico di dispersione a due dimensioni')
    plt.grid()
    if salva_dati2:
        plt.savefig('C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\results\\Immagini\\grafico_sparsi1.png')
        print("Dati salvati: il grafico è stato salvato come 'grafico_sparsi1.png'")
    plt.show()

# Calcola il tempo di esecuzione
end_time = time.time()
execution_time = end_time - start_time
print(f"Tempo di esecuzione: {execution_time} secondi")