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

# dataset
filename = "C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\dataset\\neuroblastoma.csv"
fields = []
rows = []
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        rows.append(row)

neuroblastoma_dataset = []
for row in rows:
    new_row = []
    for element in row:
        new_row.append(float(element))
    neuroblastoma_dataset.append(new_row)

filename = "C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\dataset\\cardiac_arrest.csv"
fields = []
rows = []
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        rows.append(row)

cardiac_arrest_dataset = []
for row in rows:
    new_row = []
    for element in row:
        new_row.append(float(element))
    cardiac_arrest_dataset.append(new_row)

filename = "C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\dataset\\diabetes.csv"
fields = []
rows = []
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        rows.append(row)

diabetes_dataset = []
for row in rows:
    new_row = []
    for element in row:
        new_row.append(float(element))
    diabetes_dataset.append(new_row)

filename = "C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\dataset\\sepsis.csv"
fields = []
rows = []
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        rows.append(row)

sepsis_dataset = []
for row in rows:
    new_row = []
    for element in row:
        new_row.append(float(element))
    sepsis_dataset.append(new_row)

filename = "C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\dataset\\heart.csv"
fields = []
rows = []
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        rows.append(row)

heart_dataset = []
for row in rows:
    new_row = []
    for element in row:
        new_row.append(float(element))
    heart_dataset.append(new_row)

# calcolo labels kmeans
def label_kmeans_2(matrix):
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
    kmeans.fit(matrix)
    return kmeans.labels_

def label_kmeans_3(matrix):
    kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto")
    kmeans.fit(matrix)
    return kmeans.labels_

def label_kmeans_4(matrix):
    kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto")
    kmeans.fit(matrix)
    return kmeans.labels_

def label_manhattan_2(matrix, n_clusters=2, metric='manhattan'):
    kmedoids = KMedoids(n_clusters=n_clusters, metric=metric, random_state=0)
    kmedoids.fit(matrix)
    return kmedoids.labels_

def label_manhattan_3(matrix, n_clusters=3, metric='manhattan'):
    kmedoids = KMedoids(n_clusters=n_clusters, metric=metric, random_state=0)
    kmedoids.fit(matrix)
    return kmedoids.labels_

def label_manhattan_4(matrix, n_clusters=4, metric='manhattan'):
    kmedoids = KMedoids(n_clusters=n_clusters, metric=metric, random_state=0)
    kmedoids.fit(matrix)
    return kmedoids.labels_

def label_cosine_2(matrix, n_clusters=2, metric='cosine'):
    kmedoids = KMedoids(n_clusters=n_clusters, metric=metric, random_state=0)
    kmedoids.fit(matrix)
    return kmedoids.labels_

def label_cosine_3(matrix, n_clusters=3, metric='cosine'):
    kmedoids = KMedoids(n_clusters=n_clusters, metric=metric, random_state=0)
    kmedoids.fit(matrix)
    return kmedoids.labels_

def label_cosine_4(matrix, n_clusters=4, metric='cosine'):
    kmedoids = KMedoids(n_clusters=n_clusters, metric=metric, random_state=0)
    kmedoids.fit(matrix)
    return kmedoids.labels_

# calcolo labels hierarchical
def label_hierarchical_2_ward(matrix, n_clusters=2, linkage='ward'):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(matrix)
    return labels

def label_hierarchical_3_ward(matrix, n_clusters=3, linkage='ward'):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(matrix)
    return labels

def label_hierarchical_4_ward(matrix, n_clusters=4, linkage='ward'):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(matrix)
    return labels

def label_hierarchical_2_complete(matrix, n_clusters=2, linkage='complete'):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(matrix)
    return labels

def label_hierarchical_3_complete(matrix, n_clusters=3, linkage='complete'):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(matrix)
    return labels

def label_hierarchical_4_complete(matrix, n_clusters=4, linkage='complete'):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(matrix)
    return labels

def label_hierarchical_2_average(matrix, n_clusters=2, linkage='average'):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(matrix)
    return labels

def label_hierarchical_3_average(matrix, n_clusters=3, linkage='average'):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(matrix)
    return labels

def label_hierarchical_4_average(matrix, n_clusters=4, linkage='average'):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(matrix)
    return labels

#mean-shift
def label_mean_shift(matrix):
    mean_shift = MeanShift()
    mean_shift.fit(matrix)
    return mean_shift.labels_


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

#HDBSCAN
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
    M2 = torch.tensor(filtered_X)
    labels = torch.tensor(filtered_labels)
    result = dunn_index(M2, labels).item()
    print(result)
    graf.append(result)
    return result
###############################

graf = []
M = neuroblastoma_dataset #da cambiare qui
M2 = torch.tensor(M)
dunn_index = DunnIndex(p=2)

print("\n k-means, euclidean: ")
labels = torch.tensor(label_kmeans_2(M2))
result = dunn_index(M2, labels).item()
print(result)
graf.append(result)
labels = torch.tensor(label_kmeans_3(M2))
result = dunn_index(M2, labels).item()
print(result)
graf.append(result)
labels = torch.tensor(label_kmeans_4(M2))
result = dunn_index(M2, labels).item()
print(result)
graf.append(result)


print("\n k-means, manhattan: ")
if M != neuroblastoma_dataset:
#print(label_km5(M2))
#print(label_km6(M2))
#print(label_km7(M2))
    labels = torch.tensor(label_manhattan_2(M2))
    result = dunn_index(M2, labels).item()
    print(result)
    graf.append(result)
    labels = torch.tensor(label_manhattan_3(M2))
    result = dunn_index(M2, labels).item()
    print(result)
    graf.append(result)
    labels = torch.tensor(label_manhattan_4(M2))
    result = dunn_index(M2, labels).item()
    print(result)
    graf.append(result)
else: 
    graf.append(0.6671231985092163)
    graf.append(0.42621636390686035)

print("\n k-means, cosine: ")
labels = torch.tensor(label_cosine_2(M2))
result = dunn_index(M2, labels).item()
print(result)
graf.append(result)
labels = torch.tensor(label_cosine_3(M2))
result = dunn_index(M2, labels).item()
print(result)
graf.append(result)
labels = torch.tensor(label_cosine_4(M2))
result = dunn_index(M2, labels).item()
print(result)
graf.append(result)

print("\n hierarchical, ward: ")
labels = torch.tensor(label_hierarchical_2_ward(M2))
result = dunn_index(M2, labels).item()
print(result)
graf.append(result)
labels = torch.tensor(label_hierarchical_3_ward(M2))
result = dunn_index(M2, labels).item()
print(result)
graf.append(result)
labels = torch.tensor(label_hierarchical_4_ward(M2))
result = dunn_index(M2, labels).item()
print(result)
graf.append(result)

print("\n hierarchical, complete: ")
labels = torch.tensor(label_hierarchical_2_complete(M2))
result = dunn_index(M2, labels).item()
print(result)
graf.append(result)
labels = torch.tensor(label_hierarchical_3_complete(M2))
result = dunn_index(M2, labels).item()
print(result)
graf.append(result)
labels = torch.tensor(label_hierarchical_4_complete(M2))
result = dunn_index(M2, labels).item()
print(result)
graf.append(result)

print("\n hierarchical, average: ")
labels = torch.tensor(label_hierarchical_2_average(M2))
result = dunn_index(M2, labels).item()
print(result)
graf.append(result)
labels = torch.tensor(label_hierarchical_3_average(M2))
result = dunn_index(M2, labels).item()
print(result)
graf.append(result)
labels = torch.tensor(label_hierarchical_4_average(M2))
result = dunn_index(M2, labels).item()
print(result)
graf.append(result)

print("\n DBSCAN: ")
if(M==neuroblastoma_dataset):
    dbscan(M, 1, 2) 
    dbscan(M, 3, 5)
    dbscan(M, 4, 12)
    dbscan(M, 4, 20)
if (M==sepsis_dataset):
    dbscan(M, 1, 2) 
    dbscan(M, 2, 2)
    dbscan(M, 3, 2) 
    dbscan(M, 4, 2)
if (M==cardiac_arrest_dataset):
    dbscan(M, 1, 2) 
    dbscan(M, 4, 12)
    dbscan(M, 4, 20)
if (M==diabetes_dataset):
    dbscan(M, 12, 2) 
    dbscan(M, 13, 3) 
    dbscan(M, 13, 2) 
    dbscan(M, 16, 2)
if (M==heart_dataset):
    dbscan(M, 12, 2) 
    dbscan(M, 13, 3) 
    dbscan(M, 6, 2)

def ordina_decrescente(lista):
    return sorted(lista, reverse=True)

print("\n HDBSCAN: ")
if (M==sepsis_dataset):
    hdbscan_(M, 2, 2)
    hdbscan_(M, 30, 7)
    hdbscan_(M, 50, 2) 
if (M==diabetes_dataset):
    hdbscan_(M, 2, 2)
    hdbscan_(M, 3, 2) 
if (M==neuroblastoma_dataset):
    hdbscan_(M, 5, 3)
    hdbscan_(M, 30, 7)
if (M==cardiac_arrest_dataset):
    hdbscan_(M, 2, 2)
    hdbscan_(M, 10, 7)
    hdbscan_(M, 30, 7)
if(M==heart_dataset):
    hdbscan_(M, 5, 3)
    hdbscan_(M, 50, 2) 


print("\n Mean-Shift: ")
labels = torch.tensor(label_mean_shift(M2))
result = dunn_index(M2, labels).item()
print(result)
graf.append(result)

###############################
graph = True
save_data = False 
color_dataset = 'skyblue'
title = 'nothing'
print("\n valori dunn: ", graf)
#grafico
if graph == True:
    if(M==neuroblastoma_dataset):
        color_dataset = "lightgreen"
        title = 'dataset neuroblastoma'
        etichette = ["K-M \nk=2 \nEU", "K-M \nk=3 \nEU", "K-M \nk=4 \nEU",  "K-M \nk=3 \nMAN", "K-M \nk=4 \nMAN", "K-M \nk=2 \nCOS", "K-M \nk=3 \nCOS", "K-M \nk=4 \nCOS", "HC \nk=2 \nward", "HC \nk=3 \nward", 
             "HC \nk=4 \nward", "HC \nk=2 \nCOM", "HC \nk=3 \nCOM", "HC \nk=4 \nCOM", "HC \nk=2 \nAVE", "HC \nk=3 \nAVE", "HC \nk=4 \nAVE", "DB \neps=1 \nmin=2",
             "DB \neps=3 \nmin=5", "DB \neps=4 \nmin=12", "DB \neps=4 \nmin=20","HDB \neps=5 \nmin=3", "HDB \neps=30 \nmin=7", "M-S"] 
    elif(M==cardiac_arrest_dataset):
        color_dataset = "darkred"
        title = 'dataset cardiac arrest'
        etichette = ["K-M \nk=2 \nEU", "K-M \nk=3 \nEU", "K-M \nk=4 \nEU", "K-M \nk=2 \nMAN", "K-M \nk=3 \nMAN", "K-M \nk=4 \nMAN", "K-M \nk=2 \nCOS", "K-M \nk=3 \nCOS", "K-M \nk=4 \nCOS", "HC \nk=2 \nward", "HC \nk=3 \nward", 
             "HC \nk=4 \nward", "HC \nk=2 \nCOM", "HC \nk=3 \nCOM", "HC \nk=4 \nCOM", "HC \nk=2 \nAVE", "HC \nk=3 \nAVE", "HC \nk=4 \nAVE", "DB \neps=1 \nmin=2", "DB \neps=4 \nmin=12", "DB \neps=4 \nmin=20", "HDB \neps=2 \nmin=2",
             "HDB \neps=10 \nmin=7", "HDB \neps=30 \nmin=7", "M-S"]
    elif(M==diabetes_dataset):
        title = 'dataset diabetes'
        color_dataset = "orange"
        etichette = ["K-M \nk=2 \nEU", "K-M \nk=3 \nEU", "K-M \nk=4 \nEU", "K-M \nk=2 \nMAN", "K-M \nk=3 \nMAN", "K-M \nk=4 \nMAN", "K-M \nk=2 \nCOS", "K-M \nk=3 \nCOS", "K-M \nk=4 \nCOS", "HC \nk=2 \nward", "HC \nk=3 \nward", 
             "HC \nk=4 \nward", "HC \nk=2 \nCOM", "HC \nk=3 \nCOM", "HC \nk=4 \nCOM", "HC \nk=2 \nAVE", "HC \nk=3 \nAVE", "HC \nk=4 \nAVE", "DB \neps=12 \nmin=2", "DB \neps=13 \nmin=3", "DB \neps=13 \nmin=2", "DB \neps=16 \nmin=2", "HDB \neps=2 \nmin=2", "HDB \neps=3 \nmin=2", "M-S"]
    elif(M==sepsis_dataset):
        title = 'dataset sepsis'
        color_dataset = "purple"
        etichette = ["K-M \nk=2 \nEU", "K-M \nk=3 \nEU", "K-M \nk=4 \nEU", "K-M \nk=2 \nMAN", "K-M \nk=3 \nMAN", "K-M \nk=4 \nMAN", "K-M \nk=2 \nCOS", "K-M \nk=3 \nCOS", "K-M \nk=4 \nCOS", "HC \nk=2 \nward", "HC \nk=3 \nward", 
             "HC \nk=4 \nward", "HC \nk=2 \nCOM", "HC \nk=3 \nCOM", "HC \nk=4 \nCOM", "HC \nk=2 \nAVE", "HC \nk=3 \nAVE", "HC \nk=4 \nAVE", "DB \neps=1 \nmin=2", "DB \neps=2 \nmin=2", "DB \neps=3 \nmin=2", "DB \neps=4 \nmin=2", 
             "HDB \neps=2 \nmin=2", "HDB \neps=30 \nmin=7", "HDB \neps=50 \nmin=2", "M-S"]
    elif(M==heart_dataset):
        title = 'dataset heart'
        etichette = ["K-M \nk=2 \nEU", "K-M \nk=3 \nEU", "K-M \nk=4 \nEU", "K-M \nk=2 \nMAN", "K-M \nk=3 \nMAN", "K-M \nk=4 \nMAN", "K-M \nk=2 \nCOS", "K-M \nk=3 \nCOS", "K-M \nk=4 \nCOS", "HC \nk=2 \nward", "HC \nk=3 \nward", 
             "HC \nk=4 \nward", "HC \nk=2 \nCOM", "HC \nk=3 \nCOM", "HC \nk=4 \nCOM", "HC \nk=2 \nAVE", "HC \nk=3 \nAVE", "HC \nk=4 \nAVE", "DB \neps=12 \nmin=2", "DB \neps=13 \nmin=3", "DB \neps=6 \nmin=2","HDB \neps=5 \nmin=3", "HDB \neps=50 \nmin=2", "M-S"]
    
    # Ordina i valori in ordine decrescente insieme alle etichette
    data_originated = sorted(zip(graf, etichette), key=lambda x: x[0], reverse=True)
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
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    # Sostituisci gli spazi con trattini bassi
    formatted_datetime_with_underscore = formatted_datetime.replace(' ', '_')

    #salvataggio
    if (save_data and M==neuroblastoma_dataset):
        plt.savefig('C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\results\\Immagini\\grafico_neuroblastoma.png')
        print("Dati salvati: il grafico è stato salvato come 'grafico_neuroblastoma.png'")
    elif(save_data and M==cardiac_arrest_dataset):
        plt.savefig('C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\results\\Immagini\\dataset_cardiac_arrest.png')
        print("Dati salvati: il grafico è stato salvato come 'dataset_cardiac_arrest.png'")
    elif(save_data and M==diabetes_dataset):
        plt.savefig('C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\results\\Immagini\\dataset_diabetes.png')
        print("Dati salvati: il grafico è stato salvato come 'dataset_diabetes.png'")
    elif(save_data and M==sepsis_dataset):
        plt.savefig('C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\results\\Immagini\\dataset_sepsis.png')
        print("Dati salvati: il grafico è stato salvato come 'dataset_sepsis.png'")
    elif(save_data and M==heart_dataset):
        plt.savefig('C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\results\\Immagini\\dataset_heart.png')
        print("Dati salvati: il grafico è stato salvato come 'dataset_heart.png'")
    else:
        print("Dati non salvati")
plt.tight_layout()
plt.show()

# Calcola il tempo di esecuzione
end_time = time.time()
execution_time = end_time - start_time
print(f"Tempo di esecuzione: {execution_time} secondi")

#Dataset sepsis: 4.355677604675293 secondi -> 1258 campioni 
#Dataset neuroblastoma: 3.1053531169891357 secondi -> 170 campioni
#Dataset heart: 2.908968687057495 secondi  -> 54 campioni
#Dataset cardiac arrest: 3.254814863204956 secondi -> 420 campioni
#Dataset diabetes: 3.0514817237854004 secondi  -> 68 campioni

graph_time = [[2.908968687057495, 54], [3.0514817237854004, 68], [3.1053531169891357, 170], [3.254814863204956, 420], [4.355677604675293, 1258]]
#AGGIUNGERE graph_time_after_MeanShift
# Estrazione dei dati
times = [item[0] for item in graph_time]
samples = [item[1] for item in graph_time]
# Creazione del grafico
plt.figure(figsize=(10, 6))
plt.plot(samples, times, color='black', marker='o', linestyle='-', markersize=8, alpha=1)
plt.title("Tempo di elaborazione in funzione dei datset", fontsize=14)
plt.xlabel("# punti", fontsize=12)
plt.ylabel("Tempo", fontsize=12)
plt.grid(alpha=0.4)
plt.show()