import random
from random import randint
# per intallare il pacchetto clusteval esegui "pip intstall clusteval"
# from clusteval import clusteval
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
#from random import randint
import numpy as np

#creo una matrice di 0 e 1 ordinati
matrix1= [[0,0,0,0], [0,0,0,0], [1,1,1,1], [1,1,1,1]]

#creo una matrice di 0 e 1
matrix2 = []

for i in range(4):
    n = []
    for j in range(4):
        number = randint(0,1)
        n.insert(i,number)
    matrix2.append(n)
print(matrix2)

#creo una matrice di numeri compresi fra 0 e 1
matrix3 = []

for i in range(4):
    n = []
    for j in range(4):
        number = random.uniform(0, 1)
        n.insert(i,number)
    matrix3.append(n)
print(matrix3)

#creo una matrice di numeri compresi fra 0 e 9
matrix4 = []

for i in range(4):
    n = []
    for j in range(4):
        number = random.uniform(0, 9)
        n.insert(i,number)
    matrix4.append(n)
print(matrix4)

# Applicare KMeans con k=2 ai dati della matrice matrix1
kmeans1 = KMeans(n_clusters=2)
kmeans1.fit(matrix1)
print("Centri dei cluster:", kmeans1.cluster_centers_)
print("Etichette assegnate:", kmeans1.labels_) 
etichette1 = kmeans1.labels_

# Applicare KMeans con k=2 ai dati della matrice matrix2
kmeans2 = KMeans(n_clusters=2)
kmeans2.fit(matrix2)
print("Centri dei cluster:", kmeans2.cluster_centers_)
print("Etichette assegnate:", kmeans2.labels_) #Ogni riga della matrice M è assegnata a uno dei due cluster (0 o 1).
etichette2 = kmeans2.labels_

# Applicare KMeans con k=3 ai dati della matrice matrix3
kmeans3 = KMeans(n_clusters=2)
kmeans3.fit(matrix3)
print("Centri dei cluster:", kmeans3.cluster_centers_)
print("Etichette assegnate:", kmeans3.labels_) #Ogni riga della matrice M è assegnata a uno dei due cluster (0 o 1).
etichette3 = kmeans3.labels_

# Applicare KMeans con k=4 ai dati della matrice matrix3
kmeans4 = KMeans(n_clusters=2)
kmeans4.fit(matrix4)
print("Centri dei cluster:", kmeans4.cluster_centers_)
print("Etichette assegnate:", kmeans4.labels_) #Ogni riga della matrice M è assegnata a uno dei due cluster (0 o 1).
etichette4 = kmeans4.labels_

# Inizializzare Clusteval con la metrica Dunn
#ce = clusteval(metric='dunn')
#results1 = ce.evaluate(matrix1, etichette1)
#print("risultati:", results1)


def dunn_index(data, labels):
    unique_labels = np.unique(labels)
    cluster_i = []
    cluster_j = []
    inter_cluster_distances = []
    intra_cluster_distances = []

    #Calcolare le distanze tra tutti i punti
    #dist_matrix = pairwise_distances(data)

    for i in range(len(labels)):
        if(unique_labels[0] == labels[i]):
                cluster_i.append(i)
        if(unique_labels[1] == labels[i]):
                cluster_j.append(i)
    print("cluster i", cluster_i)
    print("cluster j",cluster_j)
    # Distanze inter-cluster
    for i in range(len(cluster_i)):
         for j in range(len(cluster_j)):
              distanza = pairwise_distances([data[i]], [data[j]])[0][0]
              inter_cluster_distances.append((distanza))
              inter_cluster_distances = list(map(float, inter_cluster_distances))
    print("inter", inter_cluster_distances)
    # Distanze intra-cluster_
    for i in range(len(cluster_i)-1):
        distanza = pairwise_distances([data[i]], [data[i+1]])[0][0]
        intra_cluster_distances.append(distanza)
    for i in range(len(cluster_j)-1):
        distanza = pairwise_distances([data[j]], [data[j+1]])[0][0] #da rivedere
        intra_cluster_distances.append(distanza)
        intra_cluster_distances = list(map(float, intra_cluster_distances))
    print("intra", intra_cluster_distances)
    #metrica dunn
    return min(inter_cluster_distances) / max(intra_cluster_distances)

dunn = dunn_index(matrix1, etichette1)
print(dunn)

