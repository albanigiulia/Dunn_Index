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
M = matrix5 #da cambiare qui
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
if M != matrix1:
#print(label_km5(M2))
#print(label_km6(M2))
#print(label_km7(M2))
    labels = torch.tensor(label_km5(M2))
    result = dunn_index(M2, labels).item()
    print(result)
    graf.append(result)
    labels = torch.tensor(label_km6(M2))
    result = dunn_index(M2, labels).item()
    print(result)
    graf.append(result)
    labels = torch.tensor(label_km7(M2))
    result = dunn_index(M2, labels).item()
    print(result)
    graf.append(result)
else: 
    graf.append(0.6671231985092163)
    graf.append(0.42621636390686035)

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

if (M==matrix1 or M==matrix2 or M==matrix4):
    dbscan(M, 1, 2) 
    dbscan(M, 2, 2) 
    dbscan(M, 3, 2) 
    dbscan(M, 4, 2)
if (M==matrix1 or M==matrix2):
    dbscan(M, 3, 3)
    dbscan(M, 4, 3)
    dbscan(M, 3, 4)
    dbscan(M, 3, 5)
    dbscan(M, 4, 5)
    dbscan(M, 4, 6)
    dbscan(M, 4, 12)
    dbscan(M, 4, 20)

#funziona per file 3 e 5
if (M==matrix3 or M==matrix5):
    dbscan(M, 12, 2) 
    dbscan(M, 13, 3) 
    dbscan(M, 13, 2) 
    dbscan(M, 16, 2)
    dbscan(M, 16, 3)
    dbscan(M, 31, 2)


if (M==matrix5):
    dbscan(M, 6, 2)
    dbscan(M, 40, 2)
    dbscan(M, 50, 2)

###############################
variabile = True
salva_dati = True 
print("\n valori dunn: ", graf)
#grafico
if variabile == True:
    if(M==matrix1):
        etichette = ["K-M \nk=2 \nEU", "K-M \nk=3 \nEU", "K-M \nk=4 \nEU",  "K-M \nk=3 \nMAN", "K-M \nk=4 \nMAN", "K-M \nk=2 \nCOS", "K-M \nk=3 \nCOS", "K-M \nk=4 \nCOS", "HI \nk=2 \nward", "HI \nk=3 \nward", 
             "HI \nk=4 \nward", "HI \nk=2 \nCOM", "HI \nk=3 \nCOM", "HI \nk=4 \nCOM", "HI \nk=2 \nAVE", "HI \nk=3 \nAVE", "HI \nk=4 \nAVE", "DB \neps=1 \nmin=2", "DB \neps=2 \nmin=2", "DB \neps=3 \nmin=2", "DB \neps=4 \nmin=2",
             "DB \neps=3 \nmin=3", "DB \neps=4 \nmin=3", "DB \neps=3 \nmin=4", "DB \neps=3 \nmin=5", "DB \neps=4 \nmin=5", "DB \neps=4 \nmin=6", "DB \neps=4 \nmin=12", "DB \neps=4 \nmin=20"]   # Le etichette corrispondenti
    elif(M==matrix2):
        etichette = ["K-M \nk=2 \nEU", "K-M \nk=3 \nEU", "K-M \nk=4 \nEU", "K-M \nk=2 \nMAN", "K-M \nk=3 \nMAN", "K-M \nk=4 \nMAN", "K-M \nk=2 \nCOS", "K-M \nk=3 \nCOS", "K-M \nk=4 \nCOS", "HI \nk=2 \nward", "HI \nk=3 \nward", 
             "HI \nk=4 \nward", "HI \nk=2 \nCOM", "HI \nk=3 \nCOM", "HI \nk=4 \nCOM", "HI \nk=2 \nAVE", "HI \nk=3 \nAVE", "HI \nk=4 \nAVE", "DB \neps=1 \nmin=2", "DB \neps=2 \nmin=2", "DB \neps=3 \nmin=2", "DB \neps=4 \nmin=2",
             "DB \neps=3 \nmin=3", "DB \neps=4 \nmin=3", "DB \neps=3 \nmin=4", "DB \neps=3 \nmin=5", "DB \neps=4 \nmin=5", "DB \neps=4 \nmin=6", "DB \neps=4 \nmin=12", "DB \neps=4 \nmin=20"]
    elif(M==matrix3):
        etichette = ["K-M \nk=2 \nEU", "K-M \nk=3 \nEU", "K-M \nk=4 \nEU", "K-M \nk=2 \nMAN", "K-M \nk=3 \nMAN", "K-M \nk=4 \nMAN", "K-M \nk=2 \nCOS", "K-M \nk=3 \nCOS", "K-M \nk=4 \nCOS", "HI \nk=2 \nward", "HI \nk=3 \nward", 
             "HI \nk=4 \nward", "HI \nk=2 \nCOM", "HI \nk=3 \nCOM", "HI \nk=4 \nCOM", "HI \nk=2 \nAVE", "HI \nk=3 \nAVE", "HI \nk=4 \nAVE", "DB \neps=12 \nmin=2", "DB \neps=13 \nmin=3", "DB \neps=13 \nmin=2", "DB \neps=16 \nmin=2", "DB \neps=16 \nmin=3", "DB \neps=31 \nmin=2"]
    elif(M==matrix4):
        etichette = ["K-M \nk=2 \nEU", "K-M \nk=3 \nEU", "K-M \nk=4 \nEU", "K-M \nk=2 \nMAN", "K-M \nk=3 \nMAN", "K-M \nk=4 \nMAN", "K-M \nk=2 \nCOS", "K-M \nk=3 \nCOS", "K-M \nk=4 \nCOS", "HI \nk=2 \nward", "HI \nk=3 \nward", 
             "HI \nk=4 \nward", "HI \nk=2 \nCOM", "HI \nk=3 \nCOM", "HI \nk=4 \nCOM", "HI \nk=2 \nAVE", "HI \nk=3 \nAVE", "HI \nk=4 \nAVE", "DB \neps=1 \nmin=2", "DB \neps=2 \nmin=2", "DB \neps=3 \nmin=2", "DB \neps=4 \nmin=2"]
    elif(M==matrix5):
        etichette = ["K-M \nk=2 \nEU", "K-M \nk=3 \nEU", "K-M \nk=4 \nEU", "K-M \nk=2 \nMAN", "K-M \nk=3 \nMAN", "K-M \nk=4 \nMAN", "K-M \nk=2 \nCOS", "K-M \nk=3 \nCOS", "K-M \nk=4 \nCOS", "HI \nk=2 \nward", "HI \nk=3 \nward", 
             "HI \nk=4 \nward", "HI \nk=2 \nCOM", "HI \nk=3 \nCOM", "HI \nk=4 \nCOM", "HI \nk=2 \nAVE", "HI \nk=3 \nAVE", "HI \nk=4 \nAVE", "DB \neps=12 \nmin=2", "DB \neps=13 \nmin=3", "DB \neps=13 \nmin=2", "DB \neps=16 \nmin=2", "DB \neps=16 \nmin=3", "DB \neps=31 \nmin=2", "DB \neps=6 \nmin=2", "DB \neps=40 \nmin=2", "DB \neps=50 \nmin=2"]
    plt.figure(figsize=(20, 7))
    plt.bar(range(len(graf)), graf, color='skyblue', width=1, edgecolor='black')
    plt.xlabel('Algoritmi')
    plt.ylabel('Dunn Index')
    plt.title('Grafico a Barre dei Valori')
    plt.xticks(range(len(graf)), etichette)  # Usa le etichette al posto dei numeri sull'asse x
    plt.grid(axis='y')
    # Imposta i limiti per rimuovere il bordo a destra e a sinistra
    plt.xlim(-0.5, len(graf) - 0.5)
    # Ottimizza il layout
    plt.tight_layout()

    if (salva_dati and M==matrix1):
        plt.savefig('C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\results\\Immagini\\grafico_matrix1.png')
        print("Dati salvati: il grafico è stato salvato come 'grafico_matrix1.png'")
    elif(salva_dati and M==matrix2):
        plt.savefig('C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\results\\Immagini\\grafico_matrix2.png')
        print("Dati salvati: il grafico è stato salvato come 'grafico_matrix2.png'")
    elif(salva_dati and M==matrix3):
        plt.savefig('C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\results\\Immagini\\grafico_matrix3.png')
        print("Dati salvati: il grafico è stato salvato come 'grafico_matrix3.png'")
    elif(salva_dati and M==matrix4):
        plt.savefig('C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\results\\Immagini\\grafico_matrix4.png')
        print("Dati salvati: il grafico è stato salvato come 'grafico_matrix4.png'")
    elif(salva_dati and M==matrix5):
        plt.savefig('C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\results\\Immagini\\grafico_matrix5.png')
        print("Dati salvati: il grafico è stato salvato come 'grafico_matrix5.png'")
    else:
        print("Dati non salvati")
plt.show()