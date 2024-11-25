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
def compute_labels(matrix):
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
    kmeans.fit(matrix)
    return kmeans.labels_
###############################

# valori 
columns = 5
decimals = 2
lower_limit_first_half = 100
upper_limit_first_half = 150
lower_limit2_second_half = 1
upper_limit2_second_half = 50
rows = 100

# matrice due cluster separati
def create_cluster_matrix(n):
    if n <= 0:
        raise ValueError("n deve essere maggiore di 0")

    final_matrix = []
    half_rows = rows // 2
    for _ in range(half_rows):
        row = [round(random.uniform(lower_limit_first_half, upper_limit_first_half), decimals) for _ in range(columns)]
        final_matrix.append(row)
    for _ in range(rows - half_rows):
        row = [round(random.uniform(lower_limit2_second_half, upper_limit2_second_half), decimals) for _ in range(columns)]
        final_matrix.append(row)
    return final_matrix

clustered_matrix = create_cluster_matrix(rows)

# dunn
print("Dunn index - matrice cluster separati: ")
cluster_labels = compute_labels(clustered_matrix)
tensor_matrix = torch.tensor(clustered_matrix)
labels_tensor = torch.tensor(cluster_labels)
dunn_result = dunn_index(tensor_matrix, labels_tensor).item()
print(dunn_result)

# grafico
graph_clusters_saparated = True
if graph_clusters_saparated:
    save_clusters_separated = False
    first_cluster_values = [item[0] for item in clustered_matrix] #valore appartenenti al primo cluster
    second_cluster_values = [item[1] for item in clustered_matrix] #valore appartenenti al secondo cluster

    # Imposta i colori per i cluster: Cluster 1 = blu, Cluster 2 = rosso
    colors = ['#1f77b4' if label == 1 else '#ff7f0e' for label in cluster_labels]

    plt.scatter(first_cluster_values, second_cluster_values, s=10, color=colors, marker='o')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Grafico di dispersione a due dimensioni')
    plt.grid()

    if save_clusters_separated:
        plt.savefig('C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\results\\Images\\clustered_plot.png')
        print("Dati salvati: il grafico è stato salvato come 'clustered_plot.png'")

    plt.show()
###############################

# valori 
columns = 5
lower_limit = 0
upper_limit = 100
rows = 100

# matrice valori sparsi
def create_random_matrix(rows):
    # Creiamo una lista di n righe e 5 colonne con valori casuali da 0 a 100
    random_matrix = [[random.uniform(lower_limit, upper_limit) for _ in range(columns)] for _ in range(rows)]
    return random_matrix

random_matrix = create_random_matrix(rows)

# Creazione della lista degli indici (x)
x_values = range(rows)  # Indici per le righe

# dunn
print("Dunn index - matrice valori sparsi: ")
random_labels = compute_labels(random_matrix)
tensor_matrix = torch.tensor(random_matrix)
labels_tensor = torch.tensor(random_labels)
dunn_result = dunn_index(tensor_matrix, labels_tensor).item()
print(dunn_result)

# Grafico
graph_random = True
if graph_random:
    save_random = False
    first_cluster_values = [item[0] for item in random_matrix]
    second_cluster_values = [item[1] for item in random_matrix]

    # Imposta i colori per i cluster: Cluster 1 = blu, Cluster 2 = rosso
    colors = ['#1f77b4' if label == 1 else '#ff7f0e' for label in random_labels]

    plt.scatter(first_cluster_values, second_cluster_values, s=10, color=colors, marker='o')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Grafico di dispersione a due dimensioni')
    plt.grid()

    if save_random:
        plt.savefig('C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\results\\Images\\random_plot.png')
        print("Dati salvati: il grafico è stato salvato come 'random_plot.png'")

    plt.show()

# Calcola il tempo di esecuzione
end_time = time.time()
execution_time = end_time - start_time
print(f"Tempo di esecuzione: {execution_time} secondi")
