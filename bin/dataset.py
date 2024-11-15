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
import time

# Memorizziamo il tempo iniziale
start_time = time.time()

###############################

#dataset
filename = "C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\dataset\\neuroblastoma.csv"
fields = []
rows = []
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        rows.append(row)

dataset_neuroblastoma = []
for riga in rows:
    nuova_riga = []
    for elemento in riga:
        nuova_riga.append(float(elemento))
    dataset_neuroblastoma.append(nuova_riga)

filename = "C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\dataset\\cardiac_arrest.csv"
fields = []
rows = []
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        rows.append(row)

dataset_cardiac_arrest = []
for riga in rows:
    nuova_riga = []
    for elemento in riga:
        nuova_riga.append(float(elemento))
    dataset_cardiac_arrest.append(nuova_riga)

filename = "C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\dataset\\diabetes.csv"
fields = []
rows = []
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        rows.append(row)

dataset_diabetes = []
for riga in rows:
    nuova_riga = []
    for elemento in riga:
        nuova_riga.append(float(elemento))
    dataset_diabetes.append(nuova_riga)

filename = "C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\dataset\\sepsis.csv"
fields = []
rows = []
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        rows.append(row)

dataset_sepsis = []
for riga in rows:
    nuova_riga = []
    for elemento in riga:
        nuova_riga.append(float(elemento))
    dataset_sepsis.append(nuova_riga)

filename = "C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\dataset\\heart.csv"
fields = []
rows = []
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        rows.append(row)

dataset_heart = []
for riga in rows:
    nuova_riga = []
    for elemento in riga:
        nuova_riga.append(float(elemento))
    dataset_heart.append(nuova_riga)


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
M = dataset_neuroblastoma #da cambiare qui
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
if M != dataset_neuroblastoma:
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

if (M==dataset_neuroblastoma or M==dataset_cardiac_arrest or M==dataset_sepsis):
    dbscan(M, 1, 2) 
    dbscan(M, 2, 2) 
    dbscan(M, 3, 2) 
    dbscan(M, 4, 2)
if (M==dataset_neuroblastoma or M==dataset_cardiac_arrest):
    dbscan(M, 3, 3)
    dbscan(M, 4, 3)
    dbscan(M, 3, 4)
    dbscan(M, 3, 5)
    dbscan(M, 4, 5)
    dbscan(M, 4, 6)
    dbscan(M, 4, 12)
    dbscan(M, 4, 20)

#funziona per file 3 e 5
if (M==dataset_diabetes or M==dataset_heart):
    dbscan(M, 12, 2) 
    dbscan(M, 13, 3) 
    dbscan(M, 13, 2) 
    dbscan(M, 16, 2)
    dbscan(M, 16, 3)
    dbscan(M, 31, 2)


if (M==dataset_heart):
    dbscan(M, 6, 2)
    dbscan(M, 40, 2)
    dbscan(M, 50, 2)

def ordina_decrescente(lista):
    return sorted(lista, reverse=True)


###############################
variabile = True
salva_dati = True 
print("\n valori dunn: ", graf)
#grafico
if variabile == True:
    if(M==dataset_neuroblastoma):
        etichette = ["K-M \nk=2 \nEU", "K-M \nk=3 \nEU", "K-M \nk=4 \nEU",  "K-M \nk=3 \nMAN", "K-M \nk=4 \nMAN", "K-M \nk=2 \nCOS", "K-M \nk=3 \nCOS", "K-M \nk=4 \nCOS", "HC \nk=2 \nward", "HC \nk=3 \nward", 
             "HC \nk=4 \nward", "HC \nk=2 \nCOM", "HC \nk=3 \nCOM", "HC \nk=4 \nCOM", "HC \nk=2 \nAVE", "HC \nk=3 \nAVE", "HC \nk=4 \nAVE", "DB \neps=1 \nmin=2", "DB \neps=2 \nmin=2", "DB \neps=3 \nmin=2", "DB \neps=4 \nmin=2",
             "DB \neps=3 \nmin=3", "DB \neps=4 \nmin=3", "DB \neps=3 \nmin=4", "DB \neps=3 \nmin=5", "DB \neps=4 \nmin=5", "DB \neps=4 \nmin=6", "DB \neps=4 \nmin=12", "DB \neps=4 \nmin=20"]   # Le etichette corrispondenti
    elif(M==dataset_cardiac_arrest):
        etichette = ["K-M \nk=2 \nEU", "K-M \nk=3 \nEU", "K-M \nk=4 \nEU", "K-M \nk=2 \nMAN", "K-M \nk=3 \nMAN", "K-M \nk=4 \nMAN", "K-M \nk=2 \nCOS", "K-M \nk=3 \nCOS", "K-M \nk=4 \nCOS", "HC \nk=2 \nward", "HC \nk=3 \nward", 
             "HC \nk=4 \nward", "HC \nk=2 \nCOM", "HC \nk=3 \nCOM", "HC \nk=4 \nCOM", "HC \nk=2 \nAVE", "HC \nk=3 \nAVE", "HC \nk=4 \nAVE", "DB \neps=1 \nmin=2", "DB \neps=2 \nmin=2", "DB \neps=3 \nmin=2", "DB \neps=4 \nmin=2",
             "DB \neps=3 \nmin=3", "DB \neps=4 \nmin=3", "DB \neps=3 \nmin=4", "DB \neps=3 \nmin=5", "DB \neps=4 \nmin=5", "DB \neps=4 \nmin=6", "DB \neps=4 \nmin=12", "DB \neps=4 \nmin=20"]
    elif(M==dataset_diabetes):
        etichette = ["K-M \nk=2 \nEU", "K-M \nk=3 \nEU", "K-M \nk=4 \nEU", "K-M \nk=2 \nMAN", "K-M \nk=3 \nMAN", "K-M \nk=4 \nMAN", "K-M \nk=2 \nCOS", "K-M \nk=3 \nCOS", "K-M \nk=4 \nCOS", "HC \nk=2 \nward", "HC \nk=3 \nward", 
             "HC \nk=4 \nward", "HC \nk=2 \nCOM", "HC \nk=3 \nCOM", "HC \nk=4 \nCOM", "HC \nk=2 \nAVE", "HC \nk=3 \nAVE", "HC \nk=4 \nAVE", "DB \neps=12 \nmin=2", "DB \neps=13 \nmin=3", "DB \neps=13 \nmin=2", "DB \neps=16 \nmin=2", "DB \neps=16 \nmin=3", "DB \neps=31 \nmin=2"]
    elif(M==dataset_sepsis):
        etichette = ["K-M \nk=2 \nEU", "K-M \nk=3 \nEU", "K-M \nk=4 \nEU", "K-M \nk=2 \nMAN", "K-M \nk=3 \nMAN", "K-M \nk=4 \nMAN", "K-M \nk=2 \nCOS", "K-M \nk=3 \nCOS", "K-M \nk=4 \nCOS", "HC \nk=2 \nward", "HC \nk=3 \nward", 
             "HC \nk=4 \nward", "HC \nk=2 \nCOM", "HC \nk=3 \nCOM", "HC \nk=4 \nCOM", "HC \nk=2 \nAVE", "HC \nk=3 \nAVE", "HC \nk=4 \nAVE", "DB \neps=1 \nmin=2", "DB \neps=2 \nmin=2", "DB \neps=3 \nmin=2", "DB \neps=4 \nmin=2"]
    elif(M==dataset_heart):
        etichette = ["K-M \nk=2 \nEU", "K-M \nk=3 \nEU", "K-M \nk=4 \nEU", "K-M \nk=2 \nMAN", "K-M \nk=3 \nMAN", "K-M \nk=4 \nMAN", "K-M \nk=2 \nCOS", "K-M \nk=3 \nCOS", "K-M \nk=4 \nCOS", "HC \nk=2 \nward", "HC \nk=3 \nward", 
             "HC \nk=4 \nward", "HC \nk=2 \nCOM", "HC \nk=3 \nCOM", "HC \nk=4 \nCOM", "HC \nk=2 \nAVE", "HC \nk=3 \nAVE", "HC \nk=4 \nAVE", "DB \neps=12 \nmin=2", "DB \neps=13 \nmin=3", "DB \neps=13 \nmin=2", "DB \neps=16 \nmin=2", "DB \neps=16 \nmin=3", "DB \neps=31 \nmin=2", "DB \neps=6 \nmin=2", "DB \neps=40 \nmin=2", "DB \neps=50 \nmin=2"]
    
    # Ordina i valori in ordine decrescente insieme alle etichette
    dati_ordinati = sorted(zip(graf, etichette), key=lambda x: x[0], reverse=True)
    graf_ordinato, etichette_ordinate = zip(*dati_ordinati)
    plt.figure(figsize=(20, 7))
    plt.bar(range(len(graf_ordinato)), graf_ordinato, color='skyblue', width=1, edgecolor='black')
    plt.xlabel('Algoritmi')
    plt.ylabel('Dunn Index')
    plt.title('Grafico a Barre dei Valori Ordinati in Ordine Decrescente')

    # Imposta le etichette in orizzontale con una dimensione del font ridotta
    plt.xticks(range(len(graf_ordinato)), etichette_ordinate, rotation=0, ha='center', fontsize=10)
    plt.grid(axis='y')
    plt.xlim(-0.5, len(graf_ordinato) - 0.5)
    plt.tight_layout()
    if (salva_dati and M==dataset_neuroblastoma):
        plt.savefig('C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\results\\Immagini\\grafico_neuroblastoma.png')
        print("Dati salvati: il grafico è stato salvato come 'grafico_neuroblastoma.png'")
    elif(salva_dati and M==dataset_cardiac_arrest):
        plt.savefig('C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\results\\Immagini\\dataset_cardiac_arrest.png')
        print("Dati salvati: il grafico è stato salvato come 'dataset_cardiac_arrest.png'")
    elif(salva_dati and M==dataset_diabetes):
        plt.savefig('C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\results\\Immagini\\dataset_diabetes.png')
        print("Dati salvati: il grafico è stato salvato come 'dataset_diabetes.png'")
    elif(salva_dati and M==dataset_sepsis):
        plt.savefig('C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\results\\Immagini\\dataset_sepsis.png')
        print("Dati salvati: il grafico è stato salvato come 'dataset_sepsis.png'")
    elif(salva_dati and M==dataset_heart):
        plt.savefig('C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\results\\Immagini\\dataset_heart.png')
        print("Dati salvati: il grafico è stato salvato come 'dataset_heart.png'")
    else:
        print("Dati non salvati")
plt.show()

# Calcola il tempo di esecuzione
end_time = time.time()
execution_time = end_time - start_time
print(f"Tempo di esecuzione: {execution_time} secondi")