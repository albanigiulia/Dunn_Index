import numpy as np
from sklearn.cluster import KMeans
import random
import matplotlib.pyplot as plt
import torch
from torchmetrics.clustering import DunnIndex
import time
import datetime

# Memorizziamo il tempo iniziale
start_time = time.time()
###############################

# Calcolo labels kmeans
def compute_labels(matrix):
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
    kmeans.fit(matrix)
    return kmeans.labels_
###############################

# Creazione matrice
num_rows = int(input("Inserisci il numero di righe (deve essere pari): "))
if num_rows % 2 != 0:
    print("Il numero di righe deve essere pari.")
else:
    # Crea la lista con la prima metà di zeri e la seconda metà di uni
    data_list = [[0, 0, 0, 0] for _ in range(num_rows // 2)] + [[1, 1, 1, 1] for _ in range(num_rows // 2)]
# print(data_list)
###############################

# Sostituzioni
dunn_index_list = []
for i in range(len(data_list)):
    # Sostituisci la riga con 4 valori casuali tra 0 e 1
    data_list[i] = [random.uniform(0, 1) for _ in range(4)]
    tensor_data = torch.tensor(data_list)
    labels_tensor = torch.tensor(compute_labels(tensor_data))
    dunn_index = DunnIndex(p=2)
    result = dunn_index(tensor_data, labels_tensor).item()
    dunn_index_list.append(result)
###############################

# Grafico zeri_uni
save_data = True
x_values = range(len(dunn_index_list))
# Creazione del grafico con solo punti, senza linee
plt.plot(x_values, dunn_index_list, 'o-', color='black', markersize=3)
# Aggiunta dei titoli e delle etichette
plt.title('Grafico dei dati')
plt.xlabel('# Righe manipolate')
plt.ylabel('Dunn index')
# Imposta le etichette dell'asse x:
step_size = int(input("Inserisci il numero di step: "))
ticks = [0] + list(range(step_size - 1, len(dunn_index_list), step_size))  # Inizia con 0 e poi ogni step_size
labels = [1] + [i + 1 for i in range(step_size - 1, len(dunn_index_list), step_size)]  # Prima etichetta è 1, poi ogni step_size
plt.xticks(ticks=ticks, labels=labels) # utilizzata per impostare i valori e le etichette sull'asse x 
plt.grid()

#timestamp
current_datetime = datetime.datetime.now()
# Formatta la data e l'ora
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")+"_matrix_zero_one_"+str(num_rows)
# Sostituisci gli spazi con trattini bassi
formatted_datetime_with_underscore = formatted_datetime.replace(' ', '_')
#print(formatted_datetime_with_underscore)

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

# Tempo di esecuzione con 50 campioni: 5.246044683456421 secondi
# Tempo di esecuzione con 100 campioni: 5.266528606414795 secondi
# Tempo di esecuzione con 500 campioni: 12.036033391952515 secondi
# Tempo di esecuzione con 1000 campioni: 15.379289388656616 secondi

execution_graph_data = [[5.246044683456421, 50], [5.266528606414795, 100], [12.036033391952515, 500], [15.379289388656616, 1000]]
# Estrazione dei dati
execution_times = [item[0] for item in execution_graph_data]
sample_counts = [item[1] for item in execution_graph_data]
# Creazione del grafico
plt.figure(figsize=(10, 6))
plt.plot(sample_counts, execution_times, color='black', marker='o', linestyle='-', markersize=8, alpha=1)
plt.title("Tempo di elaborazione in funzione del numero di campioni", fontsize=14)
plt.xlabel("# punti", fontsize=12)
plt.ylabel("Tempo", fontsize=12)
plt.grid(alpha=0.4)
plt.show()
