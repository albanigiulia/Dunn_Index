import random
from random import randint
# per intallare il pacchetto clusteval esegui "pip intstall clusteval"
#from clusteval import clusteval
from sklearn.cluster import KMeans
#from random import randint
import numpy as np

#creo una matrice di 0 e 1 ordinati
matrix_zero_uno_or = [[0,0,0,0], [0,0,0,0], [1,1,1,1], [1,1,1,1]] 

#creo una matrice di 0 e 1
matrix_zero_uno = []

for i in range(4):
    n = []
    for j in range(4):
        number = randint(0,1)
        n.insert(i,number)
    matrix_zero_uno.append(n)
print(matrix_zero_uno)

#creo una matrice di numeri compresi fra 0 e 1
matrix_compresi = []

for i in range(4):
    n = []
    for j in range(4):
        number = random.uniform(0, 1)
        n.insert(i,number)
    matrix_compresi.append(n)
print(matrix_compresi)

#creo una matrice di numeri compresi fra 0 e 9
matrix3 = []

for i in range(4):
    n = []
    for j in range(4):
        number = random.uniform(0, 9)
        n.insert(i,number)
    matrix_compresi.append(n)
print(matrix_compresi)

# Applicare KMeans con k=2 ai dati della matrice matrix_zero_uno_or
kmeans = KMeans(n_clusters=2)
kmeans.fit(matrix_zero_uno_or)
print("Centri dei cluster:", kmeans.cluster_centers_)
print("Etichette assegnate:", kmeans.labels_) 

# Applicare KMeans con k=2 ai dati della matrice matrix_zero_uno
kmeans1 = KMeans(n_clusters=2)
kmeans1.fit(matrix_zero_uno)
print("Centri dei cluster:", kmeans1.cluster_centers_)
print("Etichette assegnate:", kmeans1.labels_) #Ogni riga della matrice M è assegnata a uno dei due cluster (0 o 1).


# Inizializzare Clusteval con la metrica Dunn
#ce = clusteval(metric='dunn')
#dunn_index1 = ce.fit(matrix_zero_uno)
#print(dunn_index1)
