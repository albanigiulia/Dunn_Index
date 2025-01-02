import rpy2.robjects as ro
#import dunn_R (non lo trovo) 
import numpy as np
import pandas as pd
import random
import os

os.environ['R_HOME'] = "C:/Program Files/R/R-4.4.2"  # Percorso della tua installazione di R
from dataset import filepaths;

ro.r('install.packages("clValid")')  # Installa clValid
ro.r('library(clValid)')  # Carica il pacchetto clValid

#importo i dataset
neuroblastoma_dataset = pd.read_csv(filepaths["neuroblastoma"])
#...
dataset = neuroblastoma_dataset


#SEED
#random.seed(10)
#print(random.random())


# importo le etichette
labels = [0] #???

# converto i dati in R
dataset_r = ro.r['matrix'](np.array(dataset))
labels_r = ro.IntVector(labels)

ro.r.assign('data', dataset_r)  # Assegna i dati a una variabile R
ro.r.assign('clust', labels_r)  # Assegna le etichette a una variabile R

# Calcoliamo l'indice di Dunn
dunn_index = ro.r('intCriteria')(dataset_r, labels_r, "Dunn")
print(f"Indice di Dunn: {dunn_index[0]}")


#L'errore R_HOME must be set in the environment or Registry si verifica perché l'ambiente Python, 
# tramite il modulo rpy2, non riesce a trovare l'installazione di R nel sistema
