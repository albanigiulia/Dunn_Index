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

# calcolo indice Dunn
def dunnIndex(matrix, labels):
    data = np.array(matrix)
    y_pred = np.array(labels)
    cm = ClusteringMetric(X=data, y_pred=y_pred)

    print(cm.dunn_index())

###############################
filename = "C:\\Users\\giuli\\OneDrive\\Desktop\\py\\dataset\\file1.csv"
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

filename = "C:\\Users\\giuli\\OneDrive\\Desktop\\py\\dataset\\file2_0.csv"
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

filename = "C:\\Users\\giuli\\OneDrive\\Desktop\\py\\dataset\\file3.csv"
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

filename = "C:\\Users\\giuli\\OneDrive\\Desktop\\py\\dataset\\file4.csv"
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

filename = "C:\\Users\\giuli\\OneDrive\\Desktop\\py\\dataset\\file5.csv"
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
def label(matrix):
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
    kmeans.fit(matrix)
    return kmeans.labels_

#calcolo labels hierarchical
def label_hi(matrix, n_clusters=3, linkage='average'):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(matrix)
    return labels


#######################
labels4= label(matrix4) # -> 10_7717_peerj_5665_dataYM2018_neuroblastoma -> file 1
print("label 4: ", labels4)
print("\n dunnIndex matrix4: ")
dunnIndex(matrix4, labels4)

labels5= label(matrix5) #-> journal.pone.0175818_S1Dataset_Spain_cardiac_arrest_EDITED. -> file 2
print("label 5: ", labels5)
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



############################
#matrice di zeri e uni
n = int(input("Inserisci il numero di righe (deve essere pari): "))
if n % 2 != 0:
    print("Il numero di righe deve essere pari.")
else:
    # Crea la lista con la prima metà di zeri e la seconda metà di uni
    lista = [[0, 0, 0, 0] for _ in range(n // 2)] + [[1, 1, 1, 1] for _ in range(n // 2)]
#print(lista)

#dunn index 
labels_lista= label(lista)
dunnIndex(lista, labels_lista)

#sostituzioni
lista_dunn= []
lista_dunn.append(5)
for i in range(len(lista)):
    # Sostituisci la riga con 4 valori casuali tra 0 e 1
    lista[i] = [random.uniform(0, 1) for _ in range(4)]
    #print(lista)
    #labels_lista= label(lista)
    #dunnIndex(lista, labels_lista)
    #print(lista_dunn)
    lista2 = torch.tensor(lista)
    labels_lista = torch.tensor(label(lista2))
    dunn_index = DunnIndex(p=2)
    result = dunn_index(lista2, labels_lista).item()
    lista_dunn.append(result)

#grafico righe implementate
x = range(len(lista_dunn))
# Creazione del grafico con solo punti, senza linee
plt.plot(x, lista_dunn, 'o-', color='black', markersize=3)  # 'o' specifica solo i punti
# Aggiunta dei titoli e delle etichette
plt.title('Grafico dei dati')
plt.xlabel('righe implementate')
plt.ylabel('Dunn index')
# Mostra il grafico
plt.grid()
plt.show()

############################
#matrice due cluster separati
def create_list(n):
    if n <= 0:
        raise ValueError("n deve essere maggiore di 0")

    final_list = []
    half_n = n // 2
    for _ in range(half_n):
        row = [round(random.uniform(7.5, 9.5), 2) for _ in range(5)]
        final_list.append(row)
    for _ in range(n - half_n):
        row = [round(random.uniform(0.5, 3), 2) for _ in range(5)]
        final_list.append(row)
    return final_list
n = 100 
ordinata = create_list(n)

# grafico
x = [item[0] for item in ordinata]
y = [item[1] for item in ordinata]
plt.scatter(x, y, s=10, color='black', marker='o')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Grafico di dispersione a due dimensioni')
plt.grid()
plt.show()

#dunn
labels_ordinata= label(ordinata)
dunnIndex(ordinata, labels_ordinata)


############################
#matrice valori sparsi

def crea_matrice(n):
    # Creiamo una lista di n righe e 5 colonne con valori casuali da 0 a 10
    matrice = [[random.uniform(0, 10) for _ in range(5)] for _ in range(n)]
    return matrice

n = 100
matrice_random = crea_matrice(n)
# Creazione della lista degli indici (x)
x = range(n)  # Indici per le righe

#dunn
labels_random = label(matrice_random)
dunnIndex(matrice_random, labels_random)

# grafico
x = [item[0] for item in matrice_random]
y = [item[1] for item in matrice_random]
plt.scatter(x, y, s=10, color='black', marker='o')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Grafico di dispersione a due dimensioni')
plt.grid()
plt.show()