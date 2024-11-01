import numpy as np
import random
from sklearn.cluster import KMeans
import csv
import matplotlib.pyplot as plt
import torch
from torchmetrics.clustering import DunnIndex
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn_extra.cluster import KMedoids

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

matrix1 = []
for riga in rows:
    nuova_riga = []
    for elemento in riga:
        nuova_riga.append(float(elemento))
    matrix1.append(nuova_riga)

filename = "C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\dataset\\file2_0.csv"
fields = []
rows = []
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        rows.append(row)

matrix2 = []
for riga in rows:
    nuova_riga = []
    for elemento in riga:
        nuova_riga.append(float(elemento))
    matrix2.append(nuova_riga)

filename = "C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\dataset\\file3.csv"
fields = []
rows = []
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        rows.append(row)

matrix3 = []
for riga in rows:
    nuova_riga = []
    for elemento in riga:
        nuova_riga.append(float(elemento))
    matrix3.append(nuova_riga)

filename = "C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\dataset\\file4.csv"
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

filename = "C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\dataset\\file5.csv"
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


# calcolo labels kmeans
def label_km2(matrix):
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
    kmeans.fit(matrix)
    return kmeans.labels_

def label_km3(matrix):
    kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto")
    kmeans.fit(matrix)
    return kmeans.labels_

def label_km4(matrix):
    kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto")
    kmeans.fit(matrix)
    return kmeans.labels_

def label_km5(matrix, n_clusters=2, metric='manhattan'):
    kmedoids = KMedoids(n_clusters=n_clusters, metric=metric, random_state=0)
    kmedoids.fit(matrix)
    return kmedoids.labels_

def label_km6(matrix, n_clusters=3, metric='manhattan'):
    kmedoids = KMedoids(n_clusters=n_clusters, metric=metric, random_state=0)
    kmedoids.fit(matrix)
    return kmedoids.labels_

def label_km7(matrix, n_clusters=4, metric='manhattan'):
    kmedoids = KMedoids(n_clusters=n_clusters, metric=metric, random_state=0)
    kmedoids.fit(matrix)
    return kmedoids.labels_

def label_km8(matrix, n_clusters=2, metric='cosine'):
    kmedoids = KMedoids(n_clusters=n_clusters, metric=metric, random_state=0)
    kmedoids.fit(matrix)
    return kmedoids.labels_

def label_km9(matrix, n_clusters=3, metric='cosine'):
    kmedoids = KMedoids(n_clusters=n_clusters, metric=metric, random_state=0)
    kmedoids.fit(matrix)
    return kmedoids.labels_

def label_km10(matrix, n_clusters=4, metric='cosine'):
    kmedoids = KMedoids(n_clusters=n_clusters, metric=metric, random_state=0)
    kmedoids.fit(matrix)
    return kmedoids.labels_

#calcolo labels hierarchical
def label_hi1(matrix, n_clusters=2, linkage='ward'):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(matrix)
    return labels

def label_hi2(matrix, n_clusters=3, linkage='ward'):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(matrix)
    return labels

def label_hi3(matrix, n_clusters=4, linkage='ward'):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(matrix)
    return labels

def label_hi4(matrix, n_clusters=2, linkage='complete'):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(matrix)
    return labels

def label_hi5(matrix, n_clusters=3, linkage='complete'):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(matrix)
    return labels

def label_hi6(matrix, n_clusters=4, linkage='complete'):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(matrix)
    return labels

def label_hi7(matrix, n_clusters=2, linkage='average'):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(matrix)
    return labels

def label_hi8(matrix, n_clusters=3, linkage='average'):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(matrix)
    return labels

def label_hi9(matrix, n_clusters=4, linkage='average'):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(matrix)
    return labels

#DBSCAN
def dbscan(matrix, ep, ms):
    filtered_X = []
    dunn_index = DunnIndex(p=2)
    clustering = DBSCAN(eps=ep, min_samples=ms).fit(matrix)
    lab = clustering.labels_
    #print(lab)
    for i in range(len(lab)):
        if lab[i] != -1:  # Se lab[i] non è -1, mantieni la riga
            filtered_X.append(matrix[i])
    filtered_labels = lab[lab != -1]
    #print(lab)
    M2 = torch.tensor(filtered_X)
    labels = torch.tensor(filtered_labels)
    result = dunn_index(M2, labels).item()
    print(result)
    graf.append(result)
    return result
###############################

graf = []
M = matrix4
M2 = torch.tensor(M)
dunn_index = DunnIndex(p=2)

print("\n k-means, euclidean: ")
labels = torch.tensor(label_km2(M2))
result = dunn_index(M2, labels).item()
print(result)
graf.append(result)
labels = torch.tensor(label_km3(M2))
result = dunn_index(M2, labels).item()
print(result)
graf.append(result)
labels = torch.tensor(label_km4(M2))
result = dunn_index(M2, labels).item()
print(result)
graf.append(result)


print("\n k-means, manhattan: ")
print(label_km5(M2))
print(label_km6(M2))
print(label_km7(M2))
#labels = torch.tensor(label_km5(M2))
#result = dunn_index(M2, labels).item()
#print(result)
#graf.append(result)
#labels = torch.tensor(label_km6(M2))
#result = dunn_index(M2, labels).item()
#print(result)
#graf.append(result)
#labels = torch.tensor(label_km7(M2))
#result = dunn_index(M2, labels).item()
#print(result)
#graf.append(result)

print("\n k-means, cosine: ")
labels = torch.tensor(label_km8(M2))
result = dunn_index(M2, labels).item()
print(result)
graf.append(result)
labels = torch.tensor(label_km9(M2))
result = dunn_index(M2, labels).item()
print(result)
graf.append(result)
labels = torch.tensor(label_km10(M2))
result = dunn_index(M2, labels).item()
print(result)
graf.append(result)

print("\n hierarchical, ward: ")
labels = torch.tensor(label_hi1(M2))
result = dunn_index(M2, labels).item()
print(result)
graf.append(result)
labels = torch.tensor(label_hi2(M2))
result = dunn_index(M2, labels).item()
print(result)
graf.append(result)
labels = torch.tensor(label_hi3(M2))
result = dunn_index(M2, labels).item()
print(result)
graf.append(result)

print("\n hierarchical, complete: ")
labels = torch.tensor(label_hi4(M2))
result = dunn_index(M2, labels).item()
print(result)
graf.append(result)
labels = torch.tensor(label_hi5(M2))
result = dunn_index(M2, labels).item()
print(result)
graf.append(result)
labels = torch.tensor(label_hi6(M2))
result = dunn_index(M2, labels).item()
print(result)
graf.append(result)

print("\n hierarchical, average: ")
labels = torch.tensor(label_hi7(M2))
result = dunn_index(M2, labels).item()
print(result)
graf.append(result)
labels = torch.tensor(label_hi8(M2))
result = dunn_index(M2, labels).item()
print(result)
graf.append(result)
labels = torch.tensor(label_hi9(M2))
result = dunn_index(M2, labels).item()
print(result)
graf.append(result)

print("\n DBSCAN: ")

#dbscan(M, 0.001, 2)
#dbscan(M, 0.01, 2)
#dbscan(M, 0.05, 2)
#dbscan(M, 0.1, 2)
dbscan(M, 1, 2) #funziona per file 1 2 e 4
dbscan(M, 2, 2) #funziona per file 1 2 e 4
dbscan(M, 3, 2) #funziona per file 1 2 e 4
dbscan(M, 4, 2) #funziona per file 1 2 e 4
#dbscan(M, 1, 3) #funziona per file 1
#dbscan(M, 2, 3) #funziona per file 1
#dbscan(M, 3, 3) #funziona per file 1 e 2
#dbscan(M, 4, 3) #funziona per file 1 e 2
#dbscan(M, 3, 4) #funziona per file 1 e 2
#dbscan(M, 3, 5) #funziona per file 1 e 2
#dbscan(M, 4, 5) #funziona per file 1 e 2
#dbscan(M, 4, 6) #funziona per file 1 e 2
#dbscan(M, 4, 12) #funziona per file 1 e 2
#dbscan(M, 4, 20) #funziona per file 1 e 2
###############################
variabile = False
#grafico
if variabile == True:
    print("\n valori dunn: ", graf)
    etichette = ["K-M \nk=2 \nEU", "K-M \nk=3 \nEU", "K-M \nk=4 \nEU", "K-M \nk=2 \nCOS", "K-M \nk=3 \nCOS", "K-M \nk=4 \nCOS", "HI \nk=2 \nward", "HI \nk=3 \nward", 
             "HI \nk=4 \nward", "HI \nk=2 \nCOM", "HI \nk=3 \nCOM", "HI \nk=4 \nCOM", "HI \nk=2 \nAVE", "HI \nk=3 \nAVE", "HI \nk=4 \nAVE", 
             "DB \neps=3 \nminS=2", "DB \neps=3 \nminS=5"]  # Le etichette corrispondenti
    plt.figure(figsize=(12, 7))
    plt.bar(range(len(graf)), graf, color='skyblue', width=1, edgecolor='black')
    plt.xlabel('Algoritmi')
    plt.ylabel('Dunn Index')
    plt.title('Grafico a Barre dei Valori')
    plt.xticks(range(len(graf)), etichette)  # Usa le etichette al posto dei numeri
    plt.grid(axis='y')
    # Imposta i limiti per rimuovere il bordo a destra e a sinistra
    plt.xlim(-0.5, len(graf) - 0.5)
    # Ottimizza il layout
    plt.tight_layout()
    # Mostra il grafico
    plt.tight_layout()
    plt.show()