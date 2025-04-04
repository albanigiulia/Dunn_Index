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
num_rows = 50 #qui
if num_rows % 2 != 0:
    print("Il numero di righe deve essere pari.")
else:
    # Crea una matrice con `num_rows` righe e colonne
    data_list = [[0 for _ in range(num_rows)] for _ in range(num_rows // 2)] + \
        [[1 for _ in range(num_rows)] for _ in range(num_rows // 2)]
    #print(data_list)
###############################

# Sostituzioni
dunn_index_list = []
counter = 0
available_rows = set(range(num_rows))
while counter<num_rows:
    casual_row = random.choice(list(available_rows))
    available_rows.remove(casual_row)
    counter +=1

    # Sostituisci la riga con num_rows valori casuali tra 0 e 1
    data_list[casual_row] = [random.uniform(0, 1) for _ in range(num_rows)]
    tensor_data = torch.tensor(data_list)
    labels_tensor = torch.tensor(compute_labels(tensor_data))
    dunn_index = DunnIndex(p=2)
    result = dunn_index(tensor_data, labels_tensor).item()
    dunn_index_list.append(result)

    # Stampa progressi
    print(f"Riga casuale sostituita: {casual_row}")
###############################

# Grafico zeri_uni
save_data_plot = True
x_values = range(len(dunn_index_list))
# Creazione del grafico con solo punti, senza linee
plt.plot(x_values, dunn_index_list, 'o-', color='black', markersize=3)
# Aggiunta dei titoli e delle etichette
plt.title('Grafico dei dati')
plt.xlabel('# Righe manipolate')
plt.ylabel('Dunn index')
# Imposta le etichette dell'asse x:
step_size = 5 #qui
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


save_time_plot = True
time_50_rowls = 0.7272336483001709
time_100_rowls = 1.146848201751709 
time_500_rowls = 27.960906267166138
time_1000_rowls = 185.20275497436523


execution_graph_data = [[time_50_rowls, 50], [time_100_rowls, 100], [time_500_rowls, 500], [time_1000_rowls, 1000]]
# Estrazione dei dati
execution_times = [item[0] for item in execution_graph_data]
sample_counts = [item[1] for item in execution_graph_data]
# Creazione del grafico
plt.figure(figsize=(10, 6))
plt.plot(sample_counts, execution_times, color='black', marker='o', linestyle='-', markersize=8, alpha=1)
plt.title("Tempo di elaborazione in funzione del numero di campioni", fontsize=14)
plt.xlabel("# punti", fontsize=12)
plt.ylabel("Tempo (secondi)", fontsize=12)
# Imposta le etichette sull'asse X
plt.xticks(sample_counts, labels=sample_counts)
# Limiti dinamici sull'asse Y
plt.ylim(0, max(execution_times) * 1.1)
plt.grid(alpha=0.4)

#timestamp
current_datetime = datetime.datetime.now()
# Formatta la data e l'ora
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")+"_matrix_zero_one_time"
# Sostituisci gli spazi con trattini bassi
formatted_datetime_with_underscore = formatted_datetime.replace(' ', '_')
#print(formatted_datetime_with_underscore)

if (save_time_plot):
    plt.savefig(f'..\\results\\images\\{formatted_datetime_with_underscore}.png')
    print(f"Dati salvati: il grafico è stato salvato come {formatted_datetime_with_underscore}.png")
else:
    print("Dati non salvati")
plt.show()