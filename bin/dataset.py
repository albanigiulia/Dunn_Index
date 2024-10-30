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
from sklearn_extra.cluster import KMedoids
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
M = matrix1
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
#non va

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
dbscan(M, 3, 2)
#dbscan(M, 5, 2)
dbscan(M, 3, 5)
###############################

#grafico
print("\n valori dunn: ", graf)
plt.figure(figsize=(10, 6))
plt.bar(range(len(graf)), graf, color='skyblue')
plt.xlabel('Algoritmi')
plt.ylabel('Dunn Index')
plt.title('Grafico a Barre dei Valori')
plt.xticks(range(len(graf)), range(1, len(graf) + 1))  # Assegna numeri agli indici
plt.grid(axis='y')

# Mostra il grafico
plt.tight_layout()
plt.show()