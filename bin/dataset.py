import numpy as np
import random
from permetrics import ClusteringMetric
from sklearn.cluster import KMeans
import csv
import matplotlib.pyplot as plt
import torch
from torchmetrics.clustering import DunnIndex
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
###############################

#calcolo indice Dunn
def dunnIndex(matrix, labels):
    data = np.array(matrix)
    y_pred = np.array(labels)
    cm = ClusteringMetric(X=data, y_pred=y_pred)

    print(cm.dunn_index())
###############################

#dataset
filename = "C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\dataset\\file1.csv"
fields = []
rows = []
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        rows.append(row)

matrix4 = []
for riga in rows:
    nuova_riga = []
    for elemento in riga:
        nuova_riga.append(float(elemento))
    matrix4.append(nuova_riga)

filename = "C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\dataset\\file2_0.csv"
fields = []
rows = []
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        rows.append(row)

matrix5 = []
for riga in rows:
    nuova_riga = []
    for elemento in riga:
        nuova_riga.append(float(elemento))
    matrix5.append(nuova_riga)

filename = "C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\dataset\\file3.csv"
fields = []
rows = []
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        rows.append(row)

matrix6 = []
for riga in rows:
    nuova_riga = []
    for elemento in riga:
        nuova_riga.append(float(elemento))
    matrix6.append(nuova_riga)

filename = "C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\dataset\\file4.csv"
fields = []
rows = []
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        rows.append(row)

matrix7 = []
for riga in rows:
    nuova_riga = []
    for elemento in riga:
        nuova_riga.append(float(elemento))
    matrix7.append(nuova_riga)

filename = "C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\dataset\\file5.csv"
fields = []
rows = []
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        rows.append(row)

matrix8 = []
for riga in rows:
    nuova_riga = []
    for elemento in riga:
        nuova_riga.append(float(elemento))
    matrix8.append(nuova_riga)

#DBSCAN -> cambia matrice a seconda del val da calcolare
matrix = matrix4
filtered_X = []
clustering = DBSCAN(eps=3, min_samples=2).fit(matrix)
lab = clustering.labels_
print(lab)
for i in range(len(lab)):
    if lab[i] != -1:  # Se lab[i] non è -1, mantieni la riga
        filtered_X.append(matrix[i])
filtered_labels = lab[lab != -1]
print("DBSCAN MATRIX 4")
dunnIndex(filtered_X, filtered_labels)

# calcolo labels kmeans
def label_km(matrix):
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
    kmeans.fit(matrix)
    return kmeans.labels_

#calcolo labels hierarchical
def label(matrix, n_clusters=2, linkage='ward'):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(matrix)
    return labels


#######################
labels4= label(matrix4) 
#print("label 4: ", labels4)
print("\n dunnIndex matrix4: ")
dunnIndex(matrix4, labels4)

labels5= label(matrix5) 
#print("label 5: ", labels5)
print("\n dunnIndex matrix5: ")
dunnIndex(matrix5, labels5)

labels6= label(matrix6)
#print("label 6: ", labels6)
print("\n dunnIndex matrix6: ")
dunnIndex(matrix6, labels6)

labels8= label(matrix8)
#print("label 8: ", labels8)
print("\n dunnIndex matrix8: ")
dunnIndex(matrix8, labels8)

labels7= label(matrix7)
#print("label 7: ", labels7)
print("\n dunnIndex matrix7: ")
dunnIndex(matrix7, labels7)