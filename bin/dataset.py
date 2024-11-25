import numpy as np  # da togliere
import random
from sklearn.cluster import KMeans
import csv
import matplotlib.pyplot as plt
import torch
from torchmetrics.clustering import DunnIndex
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import MeanShift
import time
import datetime
import hdbscan

# Memorizziamo il tempo iniziale
start_time = time.time()

###############################
# Caricamento dei dataset
def load_dataset(filepath):
    dataset = []
    with open(filepath, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Salta l'intestazione
        for row in csvreader:
            dataset.append([float(element) for element in row])
    return dataset

# Percorsi ai file
filepaths = {
    "neuroblastoma": "C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\dataset\\neuroblastoma.csv",
    "cardiac_arrest": "C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\dataset\\cardiac_arrest.csv",
    "diabetes": "C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\dataset\\diabetes.csv",
    "sepsis": "C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\dataset\\sepsis.csv",
    "heart": "C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\dataset\\heart.csv"
}

# Caricamento dei dataset
dataset_download = {name: load_dataset(path) for name, path in filepaths.items()}

# Accesso ai dataset individuali
neuroblastoma_dataset = dataset_download["neuroblastoma"]
cardiac_arrest_dataset = dataset_download["cardiac_arrest"]
diabetes_dataset = dataset_download["diabetes"]
sepsis_dataset = dataset_download["sepsis"]
heart_dataset = dataset_download["heart"]

###############################
# calcolo labels kmeans
def label_kmeans(matrix, clusters):
    kmeans = KMeans(n_clusters=clusters, random_state=0, n_init="auto")
    kmeans.fit(matrix)
    return kmeans.labels_

def label_KMedoids(matrix, n_clusters, metric):
    kmedoids = KMedoids(n_clusters=n_clusters, metric=metric, random_state=0)
    kmedoids.fit(matrix)
    return kmedoids.labels_

# calcolo labels hierarchical
def label_hierarchical(matrix, n_clusters, linkage):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(matrix)
    return labels

# calcolo labels mean-shift
def label_mean_shift(matrix):
    mean_shift = MeanShift()
    mean_shift.fit(matrix)
    return mean_shift.labels_

# calcolo labels DBSCAN
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
    dataset_torch = torch.tensor(filtered_X)
    labels = torch.tensor(filtered_labels)
    result = dunn_index(dataset_torch, labels).item()
    print(result)
    dunn_list.append(result)
    return result

# calcolo labels HDBSCAN
def hdbscan_(matrix, ep, ms):
    filtered_X = []
    dunn_index = DunnIndex(p=2)
    clustering = hdbscan.HDBSCAN(min_cluster_size=ep, min_samples=ms).fit(matrix)
    lab = clustering.labels_
    #print(lab)
    for i in range(len(lab)):
        if lab[i] != -1:  # Se lab[i] non è -1, mantieni la riga
            filtered_X.append(matrix[i])
    filtered_labels = lab[lab != -1]
    #print(lab)
    dataset_torch = torch.tensor(filtered_X)
    labels = torch.tensor(filtered_labels)
    result = dunn_index(dataset_torch, labels).item()
    print(result)
    dunn_list.append(result)
    return result
###############################

dunn_list = []
dataset = neuroblastoma_dataset #da cambiare qui
dataset_torch = torch.tensor(dataset)
dunn_index = DunnIndex(p=2)

print("\n k-means, euclidean: ")
result = dunn_index(dataset_torch, torch.tensor(label_kmeans(dataset_torch, 2))).item()
print(result)
dunn_list.append(result)
result = dunn_index(dataset_torch, torch.tensor(label_kmeans(dataset_torch, 3))).item()
print(result)
dunn_list.append(result)
result = dunn_index(dataset_torch, torch.tensor(label_kmeans(dataset_torch, 4))).item()
print(result)
dunn_list.append(result)

print("\n k-means, manhattan: ")
if dataset != neuroblastoma_dataset: #non funziona il dunn index perchè viene identificato un solo cluster 
    result = dunn_index(dataset_torch, torch.tensor(label_KMedoids(dataset_torch, 2, 'manhattan'))).item()
    print(result)
    dunn_list.append(result)
result = dunn_index(dataset_torch, torch.tensor(label_KMedoids(dataset_torch, 3, 'manhattan'))).item()
print(result)
dunn_list.append(result)
result = dunn_index(dataset_torch, torch.tensor(label_KMedoids(dataset_torch, 4, 'manhattan'))).item()
print(result)
dunn_list.append(result)

print("\n k-means, cosine: ")
result = dunn_index(dataset_torch, torch.tensor(label_KMedoids(dataset_torch, 2, 'cosine'))).item()
print(result)
dunn_list.append(result)
result = dunn_index(dataset_torch, torch.tensor(label_KMedoids(dataset_torch, 3, 'cosine'))).item()
print(result)
dunn_list.append(result)
result = dunn_index(dataset_torch, torch.tensor(label_KMedoids(dataset_torch, 4, 'cosine'))).item()
print(result)
dunn_list.append(result)

print("\n hierarchical, ward: ")
result = dunn_index(dataset_torch, torch.tensor(label_hierarchical(dataset_torch, 2, 'ward'))).item()
print(result)
dunn_list.append(result)
result = dunn_index(dataset_torch, torch.tensor(label_hierarchical(dataset_torch, 3, 'ward'))).item()
print(result)
dunn_list.append(result)
result = dunn_index(dataset_torch, torch.tensor(label_hierarchical(dataset_torch, 4, 'ward'))).item()
print(result)
dunn_list.append(result)

print("\n hierarchical, complete: ")
result = dunn_index(dataset_torch, torch.tensor(label_hierarchical(dataset_torch, 2, 'complete'))).item()
print(result)
dunn_list.append(result)
result = dunn_index(dataset_torch, torch.tensor(label_hierarchical(dataset_torch, 3, 'complete'))).item()
print(result)
dunn_list.append(result)
result = dunn_index(dataset_torch, torch.tensor(label_hierarchical(dataset_torch, 4, 'complete'))).item()
print(result)
dunn_list.append(result)

print("\n hierarchical, average: ")
result = dunn_index(dataset_torch, torch.tensor(label_hierarchical(dataset_torch, 2, 'average'))).item()
print(result)
dunn_list.append(result)
result = dunn_index(dataset_torch, torch.tensor(label_hierarchical(dataset_torch, 3, 'average'))).item()
print(result)
dunn_list.append(result)
result = dunn_index(dataset_torch, torch.tensor(label_hierarchical(dataset_torch, 4, 'average'))).item()
print(result)
dunn_list.append(result)

print("\n DBSCAN: ")
if(dataset==neuroblastoma_dataset):
    dbscan(dataset, 1, 2) 
    dbscan(dataset, 3, 5)
    dbscan(dataset, 4, 12)
    dbscan(dataset, 4, 20)
    print("\n HDBSCAN: ")
    hdbscan_(dataset, 5, 3)
    hdbscan_(dataset, 30, 7)
if (dataset==sepsis_dataset):
    dbscan(dataset, 1, 2) 
    dbscan(dataset, 2, 2)
    dbscan(dataset, 3, 2) 
    dbscan(dataset, 4, 2)
    print("\n HDBSCAN: ")
    hdbscan_(dataset, 2, 2)
    hdbscan_(dataset, 30, 7)
    hdbscan_(dataset, 50, 2) 
if (dataset==cardiac_arrest_dataset):
    dbscan(dataset, 1, 2) 
    dbscan(dataset, 4, 12)
    dbscan(dataset, 4, 20)
    print("\n HDBSCAN: ")
    hdbscan_(dataset, 2, 2)
    hdbscan_(dataset, 10, 7)
    hdbscan_(dataset, 30, 7)
if (dataset==diabetes_dataset):
    dbscan(dataset, 12, 2) 
    dbscan(dataset, 13, 3) 
    dbscan(dataset, 13, 2) 
    dbscan(dataset, 16, 2)
    print("\n HDBSCAN: ")
    hdbscan_(dataset, 2, 2)
    hdbscan_(dataset, 3, 2) 
if (dataset==heart_dataset):
    dbscan(dataset, 12, 2) 
    dbscan(dataset, 13, 3) 
    dbscan(dataset, 6, 2)
    print("\n HDBSCAN: ")
    hdbscan_(dataset, 5, 3)
    hdbscan_(dataset, 50, 2) 

print("\n Mean-Shift: ")
labels = torch.tensor(label_mean_shift(dataset_torch))
result = dunn_index(dataset_torch, labels).item()
print(result)
dunn_list.append(result)

###############################
#grafico
graph = True
save_data = True 
color_dataset = 'skyblue'
print("\n valori dunn: ", dunn_list)
if graph == True:
    if(dataset==neuroblastoma_dataset):
        color_dataset = "lightgreen"
        title = 'dataset neuroblastoma'
        etichette = ["K-M \nk=2 \nEU", "K-M \nk=3 \nEU", "K-M \nk=4 \nEU",  "K-M \nk=3 \nMAN", "K-M \nk=4 \nMAN", "K-M \nk=2 \nCOS", "K-M \nk=3 \nCOS", "K-M \nk=4 \nCOS", "HC \nk=2 \nward", "HC \nk=3 \nward", 
             "HC \nk=4 \nward", "HC \nk=2 \nCOM", "HC \nk=3 \nCOM", "HC \nk=4 \nCOM", "HC \nk=2 \nAVE", "HC \nk=3 \nAVE", "HC \nk=4 \nAVE", "DB \neps=1 \nmin=2",
             "DB \neps=3 \nmin=5", "DB \neps=4 \nmin=12", "DB \neps=4 \nmin=20","HDB \neps=5 \nmin=3", "HDB \neps=30 \nmin=7", "M-S"] 
    elif(dataset==cardiac_arrest_dataset):
        color_dataset = "darkred"
        title = 'dataset cardiac arrest'
        etichette = ["K-M \nk=2 \nEU", "K-M \nk=3 \nEU", "K-M \nk=4 \nEU", "K-M \nk=2 \nMAN", "K-M \nk=3 \nMAN", "K-M \nk=4 \nMAN", "K-M \nk=2 \nCOS", "K-M \nk=3 \nCOS", "K-M \nk=4 \nCOS", "HC \nk=2 \nward", "HC \nk=3 \nward", 
             "HC \nk=4 \nward", "HC \nk=2 \nCOM", "HC \nk=3 \nCOM", "HC \nk=4 \nCOM", "HC \nk=2 \nAVE", "HC \nk=3 \nAVE", "HC \nk=4 \nAVE", "DB \neps=1 \nmin=2", "DB \neps=4 \nmin=12", "DB \neps=4 \nmin=20", "HDB \neps=2 \nmin=2",
             "HDB \neps=10 \nmin=7", "HDB \neps=30 \nmin=7", "M-S"]
    elif(dataset==diabetes_dataset):
        title = 'dataset diabetes'
        color_dataset = "orange"
        etichette = ["K-M \nk=2 \nEU", "K-M \nk=3 \nEU", "K-M \nk=4 \nEU", "K-M \nk=2 \nMAN", "K-M \nk=3 \nMAN", "K-M \nk=4 \nMAN", "K-M \nk=2 \nCOS", "K-M \nk=3 \nCOS", "K-M \nk=4 \nCOS", "HC \nk=2 \nward", "HC \nk=3 \nward", 
             "HC \nk=4 \nward", "HC \nk=2 \nCOM", "HC \nk=3 \nCOM", "HC \nk=4 \nCOM", "HC \nk=2 \nAVE", "HC \nk=3 \nAVE", "HC \nk=4 \nAVE", "DB \neps=12 \nmin=2", "DB \neps=13 \nmin=3", "DB \neps=13 \nmin=2", "DB \neps=16 \nmin=2", "HDB \neps=2 \nmin=2", "HDB \neps=3 \nmin=2", "M-S"]
    elif(dataset==sepsis_dataset):
        title = 'dataset sepsis'
        color_dataset = "purple"
        etichette = ["K-M \nk=2 \nEU", "K-M \nk=3 \nEU", "K-M \nk=4 \nEU", "K-M \nk=2 \nMAN", "K-M \nk=3 \nMAN", "K-M \nk=4 \nMAN", "K-M \nk=2 \nCOS", "K-M \nk=3 \nCOS", "K-M \nk=4 \nCOS", "HC \nk=2 \nward", "HC \nk=3 \nward", 
             "HC \nk=4 \nward", "HC \nk=2 \nCOM", "HC \nk=3 \nCOM", "HC \nk=4 \nCOM", "HC \nk=2 \nAVE", "HC \nk=3 \nAVE", "HC \nk=4 \nAVE", "DB \neps=1 \nmin=2", "DB \neps=2 \nmin=2", "DB \neps=3 \nmin=2", "DB \neps=4 \nmin=2", 
             "HDB \neps=2 \nmin=2", "HDB \neps=30 \nmin=7", "HDB \neps=50 \nmin=2", "M-S"]
    elif(dataset==heart_dataset):
        title = 'dataset heart'
        etichette = ["K-M \nk=2 \nEU", "K-M \nk=3 \nEU", "K-M \nk=4 \nEU", "K-M \nk=2 \nMAN", "K-M \nk=3 \nMAN", "K-M \nk=4 \nMAN", "K-M \nk=2 \nCOS", "K-M \nk=3 \nCOS", "K-M \nk=4 \nCOS", "HC \nk=2 \nward", "HC \nk=3 \nward", 
             "HC \nk=4 \nward", "HC \nk=2 \nCOM", "HC \nk=3 \nCOM", "HC \nk=4 \nCOM", "HC \nk=2 \nAVE", "HC \nk=3 \nAVE", "HC \nk=4 \nAVE", "DB \neps=12 \nmin=2", "DB \neps=13 \nmin=3", "DB \neps=6 \nmin=2","HDB \neps=5 \nmin=3", "HDB \neps=50 \nmin=2", "M-S"]
    
    # Ordina i valori in ordine decrescente insieme alle etichette
    data_originated = sorted(zip(dunn_list, etichette), key=lambda x: x[0], reverse=True)
    decreasing_graph, decreasing_labels = zip(*data_originated)
    plt.figure(figsize=(20, 7))
    plt.bar(range(len(decreasing_graph)), decreasing_graph, color=color_dataset, width=1, edgecolor='black')
    plt.xlabel('Algoritmi')
    plt.ylabel('Dunn Index')
    plt.title(title) 

    # Imposta le etichette in orizzontale con una dimensione del font ridotta
    plt.xticks(range(len(decreasing_graph)), decreasing_labels, rotation=0, ha='center', fontsize=10)
    plt.grid(axis='y')
    plt.xlim(-0.5, len(decreasing_graph) - 0.5)
    plt.yticks(fontsize=12)  # Modifica la dimensione dei numeri sull'asse y
    plt.tight_layout()

    #timestamp
    current_datetime = datetime.datetime.now()
    # Formatta la data e l'ora
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")+"_"+title
    # Sostituisci gli spazi con trattini bassi
    formatted_datetime_with_underscore = formatted_datetime.replace(' ', '_')
    #print(formatted_datetime_with_underscore)

    #salvataggio
    if (save_data):
        plt.savefig(f'C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\results\\Immagini\\{formatted_datetime_with_underscore}.png')
        print(f"Dati salvati: il grafico è stato salvato come {formatted_datetime_with_underscore}")
    else:
        print("Dati non salvati")

plt.show()
# Calcola il tempo di esecuzione
end_time = time.time()
execution_time = end_time - start_time
print(f"Tempo di esecuzione: {execution_time} secondi")