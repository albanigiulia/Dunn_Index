import random
from random import randint
# per intallare il pacchetto clusteval esegui "pip intstall clusteval"
from clusteval import clusteval
from sklearn.cluster import KMeans
#from random import randint
import numpy as np

#creo una matrice di 0 e 1
matrix_zero_uno = []

for i in range(10):
    n = []
    for j in range(10):
        number = randint(0,1)
        n.insert(i,number)
    matrix_zero_uno.append(n)
print(matrix_zero_uno)

#creo una matrice di numeri compresi fra 0 e 1
matrix_compresi = []

for i in range(10):
    n = []
    for j in range(10):
        number = random.uniform(0, 1)
        n.insert(i,number)
    matrix_compresi.append(n)
print(matrix_compresi)

#creo una matrice di numeri compresi fra 0 e 9
matrix3 = []

for i in range(10):
    n = []
    for j in range(10):
        number = random.uniform(0, 9)
        n.insert(i,number)
    matrix_compresi.append(n)
print(matrix_compresi)

# Applicare KMeans con k=2 ai dati della matrice
kmeans = KMeans(n_clusters=2)
labels = kmeans.fit_predict(matrix_zero_uno)
# labels conterrà il cluster assegnato a ciascun punto
print(labels)

# Inizializzare Clusteval con la metrica Dunn
ce = clusteval(metric='dunn')
#dunn_index1 = ce.fit(matrix_zero_uno)
#print(dunn_index1)
