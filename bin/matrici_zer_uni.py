import numpy as np
from sklearn.cluster import KMeans
import random
import matplotlib.pyplot as plt
import torch
from torchmetrics.clustering import DunnIndex
import time
# Memorizziamo il tempo iniziale
start_time = time.time()
###############################

# calcolo labels kmeans
def label(matrix):
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
    kmeans.fit(matrix)
    return kmeans.labels_
###############################

#creazione matrice
n = int(input("Inserisci il numero di righe (deve essere pari): "))
if n % 2 != 0:
    print("Il numero di righe deve essere pari.")
else:
    # Crea la lista con la prima metà di zeri e la seconda metà di uni
    lista = [[0, 0, 0, 0] for _ in range(n // 2)] + [[1, 1, 1, 1] for _ in range(n // 2)]
#print(lista)
###############################

#sostituzioni
lista_dunn= []
for i in range(len(lista)):
    # Sostituisci la riga con 4 valori casuali tra 0 e 1
    lista[i] = [random.uniform(0, 1) for _ in range(4)]
    lista2 = torch.tensor(lista)
    labels_lista = torch.tensor(label(lista2))
    dunn_index = DunnIndex(p=2)
    result = dunn_index(lista2, labels_lista).item()
    lista_dunn.append(result)
###############################

#grafico zeri_uni
salva_dati = False
x = range(len(lista_dunn))
# Creazione del grafico con solo punti, senza linee
plt.plot(x, lista_dunn, 'o-', color='black', markersize=3)
# Aggiunta dei titoli e delle etichette
plt.title('Grafico dei dati')
plt.xlabel('# Righe manipolate')
plt.ylabel('Dunn index')
# Imposta le etichette dell'asse x:
step = int(input("Inserisci il numero di step: "))
ticks = [0] + list(range(step-1, len(lista_dunn), step))  # Inizia con 0 e poi ogni step
labels = [1] + [i + 1 for i in range(step-1, len(lista_dunn), step)]  # Prima etichetta è 1, poi ogni step
plt.xticks(ticks=ticks, labels=labels)
# Mostra il grafico
plt.grid()
if (salva_dati):
    plt.savefig('C:\\Users\\giuli\\OneDrive\\Desktop\\DunnIndex\\results\\Immagini\\grafico_zeri_uni.png')
    print("Dati salvati: il grafico è stato salvato come 'grafico_zeri_uni.png'")
plt.show()

# Calcola il tempo di esecuzione
end_time = time.time()
execution_time = end_time - start_time
print(f"Tempo di esecuzione: {execution_time} secondi")


#Tempo di esecuzione con 50 campioni: 5.246044683456421 secondi
#Tempo di esecuzione con 100 campioni: 5.266528606414795 secondi
#Tempo di esecuzione con 500 campioni: 12.036033391952515 secondi
#Tempo di esecuzione con 1000 campioni: 15.379289388656616 secondi

graph_time = [[5.246044683456421, 50], [5.266528606414795, 100], [12.036033391952515, 500], [15.379289388656616, 1000]]
# Estrazione dei dati
times = [item[0] for item in graph_time]
samples = [item[1] for item in graph_time]
# Creazione del grafico
plt.figure(figsize=(10, 6))
plt.plot(samples, times, color='black', marker='o', linestyle='-', markersize=8, alpha=1)
plt.title("Tempo di elaborazione in funzione del numero di campioni", fontsize=14)
plt.xlabel("# punti", fontsize=12)
plt.ylabel("Tempo", fontsize=12)
plt.grid(alpha=0.4)
plt.show()