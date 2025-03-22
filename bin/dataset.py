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
from sklearn.cluster import Birch
from sklearn.cluster import estimate_bandwidth
import time
import datetime
import hdbscan
import logging
import sys
import matplotlib.lines as mlines


#timestamp
current_datetime = datetime.datetime.now()
# Formatta la data e l'ora
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")+"_test_log"
# Sostituisci gli spazi con trattini bassi
formatted_datetime_with_underscore = formatted_datetime.replace(' ', '_')
#print(formatted_datetime_with_underscore)

# Configura il logging per scrivere nel file di log
logging.basicConfig(
    filename=(f'..\\results\\log\\{formatted_datetime_with_underscore}.txt'),
    level=logging.INFO,  # Usa INFO per evitare messaggi di debug
    format='%(asctime)s - %(message)s',  # Formato semplificato
    datefmt='%Y-%m-%d %H:%M:%S',
    filemode='a'
)
# Funzione per reindirizzare il print verso il file di log e la console
class LogToFileAndConsole:
    def write(self, message):
        if message != '\n':  # Ignora le righe vuote
            logging.info(message.strip())  # Scrive nel file di log
            sys.__stdout__.write(message)  # Ripristina la stampa sulla console
    def flush(self):
        pass  # Necessario per evitare errori di buffering

# Reindirizza sys.stdout (la stampa sulla console) al LogToFileAndConsole
sys.stdout = LogToFileAndConsole()



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
    "neuroblastoma": "..\\dataset\\neuroblastoma.csv",
    "cardiac_arrest": "..\\dataset\\cardiac_arrest.csv",
    "diabetes": "..\\dataset\\diabetes.csv",
    "sepsis": "..\\dataset\\sepsis.csv",
    "heart": "..\\dataset\\heart_failure.csv"
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
def label_mean_shift(matrix, bandwidth, bin_seeding, min_bin_freq=1, cluster_all=True):
    mean_shift = MeanShift(bandwidth = bandwidth, bin_seeding = bin_seeding, min_bin_freq=min_bin_freq, cluster_all=cluster_all)
    mean_shift.fit(matrix)
    return mean_shift.labels_

# calcolo labels birch
def label_birch(matrix, n_clusters):
    model = Birch(n_clusters=n_clusters)
    labels = model.fit_predict(matrix)
    return labels

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
    unique_numbers = set() #Set è una struttura dati che non consente duplicati
    for number in lab:
        unique_numbers.add(number)  # Aggiungo il numero al set (se non è già presente)
    print("Ci sono", len(unique_numbers), "k cluster.")
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
    unique_numbers = set() #Set è una struttura dati che non consente duplicati
    for number in lab:
        unique_numbers.add(number)  # Aggiungo il numero al set (se non è già presente)
    print("Ci sono", len(unique_numbers), "k cluster.")
    dataset_torch = torch.tensor(filtered_X)
    labels = torch.tensor(filtered_labels)
    result = dunn_index(dataset_torch, labels).item()
    print(result)
    dunn_list.append(result)
    return result
###############################

datasets = [neuroblastoma_dataset, diabetes_dataset, cardiac_arrest_dataset, sepsis_dataset, heart_dataset] #QUI
for dataset in datasets:
    dunn_list = []
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
        dbscan(dataset, 4, 20)
        print("\n HDBSCAN: ")
        hdbscan_(dataset, 2, 2)
        hdbscan_(dataset, 10, 7)
        hdbscan_(dataset, 30, 7)
    if (dataset==diabetes_dataset):
        dbscan(dataset, 12, 2) 
        dbscan(dataset, 13, 3) 
        dbscan(dataset, 13, 2) 
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
    result = dunn_index(dataset_torch, torch.tensor(label_mean_shift(dataset_torch, bandwidth=None, bin_seeding=False))).item()
    print(result)
    dunn_list.append(result)
    # Calcola il bandwidth
    calculated_bandwidth = estimate_bandwidth(dataset_torch.numpy(), quantile=0.2, n_samples=500)
    result = dunn_index(dataset_torch, torch.tensor(label_mean_shift(dataset_torch, bandwidth=calculated_bandwidth, bin_seeding=False))).item()
    print(result)
    dunn_list.append(result)
    if(dataset!=diabetes_dataset):
        result = dunn_index(dataset_torch, torch.tensor(label_mean_shift(dataset_torch, bandwidth=calculated_bandwidth, bin_seeding=True))).item()
        print(result)
        dunn_list.append(result)

    print("\n Birch: ")
    result = dunn_index(dataset_torch, torch.tensor(label_birch(dataset_torch, 2))).item()
    print(result)
    dunn_list.append(result)
    result = dunn_index(dataset_torch, torch.tensor(label_birch(dataset_torch, 3))).item()
    print(result)
    dunn_list.append(result)
    result = dunn_index(dataset_torch, torch.tensor(label_birch(dataset_torch, 4))).item()
    print(result)
    dunn_list.append(result)

    ###############################
    #grafico
    graph = True
    save_data_plot = True
    color_dataset = 'skyblue'
    print("\n valori dunn: ", dunn_list)
    if graph == True:
        if(dataset==neuroblastoma_dataset):
            position_indices = [0, 15, 20, 22, 25, 28]
            color_dataset = "lightgreen"
            title = 'Dataset neuroblastoma'
            etichette = ["K-Means \nk=2 \nd=Euclidean", "K-Means \nk=3 \nEU", "K-Means\nk=4 \nEU",  "K-Means \nk=3 \nMAN", "K-Means \nk=4 \nMAN", "K-Means \nk=2 \nCOS", "K-Means \nk=3 \nCOS", "K-Means \nk=4 \nCOS", "HC \nk=2 \nward", "HC \nk=3 \nward", 
                "HC \nk=4 \nward", "HC \nk=2 \nCOM", "HC \nk=3 \nCOM", "HC \nk=4 \nCOM", "HC \nk=2 \nAVE", "Hierarchical C.\nk=3 \nl=Average", "HC \nk=4 \nAVE", "DB \neps=1 \nmin=2",
                "DB \neps=3 \nmin=5", "DB \neps=4 \nmin=12", "DBSCAN \nEpsilon=4 \nMin Points=20","HDB \neps=5 \nmin=3", "HDBSCAN \nEpsilon=30 \nMin Points=7","M-S \nbw=NONE \nbs=False", "Mean-Shift \nbw=estimate \nbs=False", "Mean-Shift \nbw=Estimate \nbs=True","Birch \nk=2", "Birch \nk=3", "Birch \nk=4"]
        elif(dataset==cardiac_arrest_dataset):
            position_indices = [2, 12, 18, 22, 25, 27]
            color_dataset = "darkred"
            title = 'Dataset arresto cardiaco'
            etichette = ["K-Means \nk=2 \nEU", "K-M \nk=3 \nEU", "K-Means \nk=4 \nd=Euclidean", "K-M \nk=2 \nMAN", "K-M \nk=3 \nMAN", "K-M \nk=4 \nMAN", "K-M \nk=2 \nCOS", "K-M \nk=3 \nCOS", "K-M \nk=4 \nCOS", "HC \nk=2 \nward", "HC \nk=3 \nward", 
                "HC \nk=4 \nward", "Hierarchical C. \nk=2 \nl=Complete", "HC \nk=3 \nCOM", "HC \nk=4 \nCOM", "HC \nk=2 \nAVE", "HC \nk=3 \nAVE", "HC \nk=4 \nAVE", "DBSCAN \nEpsilon=1 \nMin Points=2","DB \neps=4 \nmin=20", "HDB \neps=2 \nmin=2",
                "HDB \neps=10 \nmin=7", "HDBSCAN \nEpsilon=30 \nMin Points=7", "M-S \nbw=NONE \nbs=False", "M-S \nbw=estimate \nbs=False", "Mean-Shift \nbw=Estimate \nbs=True", "Birch \nk=2", "Birch \nk=3", "Birch \nk=4"]
        elif(dataset==diabetes_dataset):
            position_indices = [0, 16, 18, 22, 23, 27]
            title = 'Dataset diabete'
            color_dataset = "orange"
            etichette = ["K-M \nkeans=2 \nd=Euclidean", "K-M \nk=3 \nEU", "K-M \nk=4 \nEU", "K-M \nk=2 \nMAN", "K-M \nk=3 \nMAN", "K-M \nk=4 \nMAN", "K-M \nk=2 \nCOS", "K-M \nk=3 \nCOS", "K-M \nk=4 \nCOS", "HC \nk=2 \nward", "HC \nk=3 \nward", 
                "HC \nk=4 \nward", "HC \nk=2 \nCOM", "HC \nk=3 \nCOM", "HC \nk=4 \nCOM", "HC \nk=2 \nAVE", "Hierarchical C. \nk=3 \nl=Average", "HC \nk=4 \nAVE", "DBSCAN \nEpsilon=12 \nMin Points=2", "DB \neps=13 \nmin=3", "DB \neps=13 \nmin=2","HDB \neps=2 \nmin=2", "HDBSCAN \nEpsilon=3 \nMin Points=2", 
                "Mean-Shift \nbw=NONE \nbs=False", "M-S \nbw=estimate \nbs=False", "M-S \nbw=estimate \nbs=True", "Birch \nk=2", "Birch \nk=3", "Birch \nk=4"]
        elif(dataset==sepsis_dataset):
            position_indices = [0, 17, 19, 23, 27, 28]
            title = 'Dataset sepsi'
            color_dataset = "purple"
            etichette = ["K-Means \nk=2 \nd=Euclidean", "K-M \nk=3 \nEU", "K-M \nk=4 \nEU", "K-M \nk=2 \nMAN", "K-M \nk=3 \nMAN", "K-M \nk=4 \nMAN", "K-M \nk=2 \nCOS", "K-M \nk=3 \nCOS", "K-M \nk=4 \nCOS", "HC \nk=2 \nward", "HC \nk=3 \nward", 
                "HC \nk=4 \nward", "HC \nk=2 \nCOM", "HC \nk=3 \nCOM", "HC \nk=4 \nCOM", "HC \nk=2 \nAVE", "HC \nk=3 \nAVE", "Hierarchical C. \nk=4 \nl=Average", "DB \neps=1 \nmin=2", "DBSCAN \nEpsilon=2 \nMin Points=2", "DB \neps=3 \nmin=2", "DB \neps=4 \nmin=2", 
                "HDBSCAN \nEpsilon=2 \nMin Points=2", "HDBSCAN \nEpsilon=30 \nMin Points=7", "HDB \neps=50 \nmin=2", "M-S \nbv=NONE \nbs=False", "M-S \nbv=estimate \nbs=False", "Mean-Shift \nbw=Estimate \nbs=True", "Birch \nk=2", "Birch \nk=3", "Birch \nk=4"]
        elif(dataset==heart_dataset):
            position_indices = [1, 12, 20, 22, 24, 26]
            title = 'Dataset insufficienza cardiaca'
            etichette = ["K-M \nk=2 \nEU", "K-Means \nk=3 \nd=Euclidean", "K-M \nk=4 \nEU", "K-M \nk=2 \nMAN", "K-M \nk=3 \nMAN", "K-M \nk=4 \nMAN", "K-M \nk=2 \nCOS", "K-M \nk=3 \nCOS", "K-M \nk=4 \nCOS", "HC \nk=2 \nward", "HC \nk=3 \nward", 
                "HC \nk=4 \nward", "Hierarchical C. \nk=2 \nl=Complete", "HC \nk=3 \nCOM", "HC \nk=4 \nCOM", "HC \nk=2 \nAVE", "HC \nk=3 \nAVE", "HC \nk=4 \nAVE", "DB \neps=12 \nmin=2", "DB \neps=13 \nmin=3", "DBSCAN \nEpsilon=6 \nMin Points=2","HDB \neps=5 \nmin=3", "HDBSCAN \nEpsilon=50 \nMin Points=2", 
                "M-S \nbw=NONE \nbs=False", "Mean-Shift \nbw=Estimate \nbs=False", "M-S \nbw=Estimate \nbs=True", "Birch \nk=2", "Birch \nk=3", "Birch \nk=4"]
        
    
        selected_data = [(dunn_list[i], etichette[i]) for i in position_indices] 
        sorted_data = sorted(selected_data, key=lambda x: x[0], reverse=True)
        sorted_graph, sorted_labels = zip(*sorted_data)

        # Crea il grafico
        plt.figure(figsize=(10, 6))

        # Usa bar per un grafico a barre verticali
        bars = plt.bar(sorted_labels, sorted_graph, color=color_dataset, edgecolor='black', width=0.6)

        # Etichette degli assi
        plt.ylabel('Dunn Index', fontsize=14)
        plt.xlabel('Algoritmi di clustering e relativi iperparametri', fontsize=14)

        # Titolo del grafico
        plt.title(title, fontsize=15)

        # Aggiungi la griglia per l'asse y
        plt.grid(axis='y')

        # Imposta i limiti per l'asse y con un margine extra per evitare che il valore esca dal grafico
        plt.ylim(0, max(sorted_graph) * 1.1)  # Aggiungi un 10% di margine extra sopra

        # Modifica la dimensione delle etichette sull'asse x
        plt.xticks(fontsize=13, rotation=0, ha='center')

        # Modifica la dimensione dei numeri sull'asse y
        plt.yticks(fontsize=13) 

        # Aggiungi i numeri sopra le barre
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f'{height:.2f}', va='bottom', ha='center', fontsize=12)


        # Posizioni dei pallini proporzionali al massimo valore sull'asse Y
        x_scatter = []
        y_scatter = []

        # Determina il massimo valore del grafico
        y_max = max(sorted_graph)  # Calcola il massimo valore delle colonne

        # Calcola le posizioni dinamiche dei pallini
        for i in range(len(sorted_labels) - 1):  # Cicla tra le colonne
            x_middle = (i + i + 1) / 2  # Posizione X centrale tra due colonne
            x_scatter.extend([x_middle - 0.10, x_middle, x_middle + 0.10])  # Posizioni X equidistanti
            y_scatter.extend([y_max * 0.01, y_max * 0.01, y_max * 0.01])  # Altezza proporzionale al massimo valore

        # Definizione della legenda con descrizioni extra
        legend_elements = [
            mlines.Line2D([], [], color="white", markersize=0, linestyle="None", label="●●●: valori di Dunn Index che non sono stati inseriti nel grafico"),
            mlines.Line2D([], [], color="white", markersize=0, linestyle="None", label="k: numero di cluster"),
            mlines.Line2D([], [], color="white", markersize=0, linestyle="None", label="bw: Bandwidth"),
            mlines.Line2D([], [], color="white", markersize=0, linestyle="None", label="bs: Bin Seeding"),
            mlines.Line2D([], [], color="white", markersize=0, linestyle="None", label="l: Linkage"),
            mlines.Line2D([], [], color="white", markersize=0, linestyle="None", label="d: metrica di distanza")
            ]
            
        # Disegna i pallini sul grafico
        plt.scatter(x_scatter, y_scatter, color="black",zorder=5)

        # Completa il resto del grafico
        plt.legend(handles=legend_elements, fontsize=12)
        plt.tight_layout()



        #timestamp
        current_datetime = datetime.datetime.now()
        # Formatta la data e l'ora
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")+"_"+title
        # Sostituisci gli spazi con trattini bassi
        formatted_datetime_with_underscore = formatted_datetime.replace(' ', '_')

        #salvataggio
        if (save_data_plot):
            plt.savefig(f'..\\results\\images\\{formatted_datetime_with_underscore}.png')
            print(f"Dati salvati: il grafico è stato salvato come {formatted_datetime_with_underscore}.png")
        else:
            print("Dati non salvati")

# Calcola il tempo di esecuzione
end_time = time.time()
execution_time = end_time - start_time
print(f"Tempo di esecuzione: {execution_time} secondi")
plt.show()

graph_time = [[0.5378162860870361, 54], [0.4418458938598633, 68], [0.4759237766265869, 170], [0.5769999027252197, 420], [1.352464199066162, 1258]]
graph_time_after_MeanShift = [[1.1574926376342773, 54], [1.399263858795166, 68], [0.7626731395721436, 170], [2.0230109691619873, 420], [92.08143591880798, 1258]] #NB

# Estrazione dei dati
times = [item[0] for item in graph_time_after_MeanShift]
samples = [item[1] for item in graph_time_after_MeanShift]
# Creazione del grafico
plt.figure(figsize=(10, 6))
plt.plot(samples, times, color='black', marker='o', linestyle='-', markersize=8, alpha=1)
plt.title("Tempo di elaborazione in funzione dei datset", fontsize=14)
plt.xlabel("# punti", fontsize=12)
plt.ylabel("Tempo (secondi)", fontsize=12)
plt.grid(alpha=0.4)

#timestamp
current_datetime = datetime.datetime.now()
# Formatta la data e l'ora
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")+"_dataset_time"
# Sostituisci gli spazi con trattini bassi
formatted_datetime_with_underscore = formatted_datetime.replace(' ', '_')
#print(formatted_datetime_with_underscore)

save_time_plot = False
if (save_time_plot):
    plt.savefig(f'..\\results\\images\\{formatted_datetime_with_underscore}.png')
    print(f"Dati salvati: il grafico è stato salvato come ..\\results\\images\\{formatted_datetime_with_underscore}.png")
else:
    print("Dati non salvati")
plt.show()