import numpy as np
from permetrics import ClusteringMetric
from sklearn.cluster import KMeans
import random
import matplotlib.pyplot as plt
import torch
from torchmetrics.clustering import DunnIndex
###############################

# calcolo indice Dunn
def dunnIndex(matrix, labels):
    data = np.array(matrix)
    y_pred = np.array(labels)
    cm = ClusteringMetric(X=data, y_pred=y_pred)

    print(cm.dunn_index())
###############################

# calcolo labels kmeans
def label(matrix):
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
    kmeans.fit(matrix)
    return kmeans.labels_
###############################

#creazione matrice
n = int(input("Inserisci il numero di righe (deve essere pari): "))
if n % 2 != 0:
    print("Il numero di righe deve essere pari.")
else:
    # Crea la lista con la prima metà di zeri e la seconda metà di uni
    lista = [[0, 0, 0, 0] for _ in range(n // 2)] + [[1, 1, 1, 1] for _ in range(n // 2)]
#print(lista)
###############################

#sostituzioni
lista_dunn= []
for i in range(len(lista)):
    # Sostituisci la riga con 4 valori casuali tra 0 e 1
    lista[i] = [random.uniform(0, 1) for _ in range(4)]
    lista2 = torch.tensor(lista)
    labels_lista = torch.tensor(label(lista2))
    dunn_index = DunnIndex(p=2)
    result = dunn_index(lista2, labels_lista).item()
    lista_dunn.append(result)
###############################

#grafico righe implementate
x = range(len(lista_dunn))
# Creazione del grafico con solo punti, senza linee
plt.plot(x, lista_dunn, 'o-', color='black', markersize=3)  # 'o' specifica solo i punti
# Aggiunta dei titoli e delle etichette
plt.title('Grafico dei dati')
plt.xlabel('# Righe manipolate')
plt.ylabel('Dunn index')
# Imposta le etichette dell'asse x per mostrare 1, 100, 200, 300, ecc.
step = int(input("Inserisci il numero di step: "))
ticks = [0] + list(range(step-1, len(lista_dunn), step))  # Inizia con 0 e poi ogni step
labels = [1] + [i + 1 for i in range(step-1, len(lista_dunn), step)]  # Prima etichetta è 1, poi ogni step
plt.xticks(ticks=ticks, labels=labels)
# Mostra il grafico
plt.grid()
plt.show()