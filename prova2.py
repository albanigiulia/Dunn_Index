import numpy as np
from permetrics import ClusteringMetric
from sklearn.cluster import KMeans
import csv

filename = "file1.csv"
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

filename = "file2_0.csv"
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

filename = "file3.csv"
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

filename = "file4.csv"
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

filename = "file5.csv"
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


# matrici di prova
matrix1= [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
matrix2= [[0.21388325, 0.4144178, 0.28142388, 0.26757954, 0.07112094, 0.57955653, 0.32419943, 0.55047146, 0.07027437, 0.87184044], [0.54800552, 0.8818611, 0.22617232, 0.86823141, 0.77470886, 0.64854027, 0.74817748, 0.21902852, 0.77650441, 0.56808367], [0.50317109, 0.57441178, 0.20083909, 0.46151987, 0.9970595, 0.88703719, 0.91362757, 0.86353683, 0.03709757, 0.16799969], [0.72483402, 0.09408283, 0.60255517, 0.77500336, 0.21148459, 0.71625775, 0.87799598, 0.40331332, 0.59270708, 0.78170712], [0.04589047, 0.32425334, 0.63269146, 0.8853751, 0.96157003, 0.81085432, 0.28864377, 0.44891262, 0.52396857, 0.6203082], [0.17799906, 0.27933984, 0.24631808, 0.23210637, 0.47319227, 0.01437758, 0.02505056, 0.67316872, 0.137065, 0.5973446], [0.42710921, 0.33238537, 0.29805439, 0.2642445, 0.34971675, 0.68122529, 0.70620061, 0.10994543, 0.35380754, 0.05460602], [0.11901042, 0.51525872, 0.13752803, 0.8222068, 0.77039247, 0.54365119, 0.48568859, 0.20044207, 0.28556211, 0.74968852], [0.12298581, 0.96098801, 0.49955447, 0.69079199, 0.6678223, 0.01212421, 0.97997341, 0.70392177, 0.78232717, 0.00627235], [0.70235684, 0.36782346, 0.09170701, 0.23006601, 0.18703349, 0.71849901, 0.54374917, 0.84611345, 0.42695674, 0.85385115]]
matrix3= [[3.74718949, 8.57980992, 8.33612174, 2.24941705, 3.97734191, 2.80719002, 1.13434965, 0.62767605, 7.05488068, 7.44582041], [8.98407709, 6.57692735, 0.74987844, 0.62396237, 8.16696039, 1.27054225, 1.31653922, 3.14421934, 4.19662256, 5.58622101], [2.3779623, 8.07327974, 7.36511329, 4.81298064, 6.26564535, 3.87286032, 6.26788002, 0.08721363, 5.05394633, 4.94100317], [0.34439589, 5.1575472, 2.91156778, 3.62214774, 4.00631184, 4.40533365, 8.2546896, 5.66226568, 1.17094321, 0.35922704], [3.60392746, 0.36753705, 3.51631296, 5.56176371, 6.98470009, 3.55856618, 3.97100222, 1.07069995, 2.90508035, 5.19945911], [6.16552085, 2.21493268, 3.69164725, 7.85615827, 5.98450127, 7.78274251, 3.2875091, 8.36471928, 4.55298355, 5.92741601], [8.43304825, 0.90204463, 8.31415437, 3.74032988, 3.00049359, 4.36403853, 8.34677409, 7.57684567, 7.63745802, 0.70836763], [1.57856913, 7.1216673, 7.29137239, 5.38744536, 6.72687499, 0.73532461, 8.96876779, 1.27527684, 3.74212572, 4.01823209], [1.98113947, 6.21764432, 2.44199443, 6.25964397, 8.43640345, 4.38429091, 2.782805, 6.33295204, 3.42129031, 4.04638331], [0.51058352, 2.37364535, 7.7202888, 1.60910123, 7.25503605, 2.08935057, 0.67052188, 3.12265423, 0.65891039, 8.88825156]]
caso = [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]]

# calcolo labels
def label(matrix):
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
    kmeans.fit(matrix)
    return kmeans.labels_

# calcolo indice Dunn
def dunnIndex(matrix, labels):
    data = np.array(matrix)
    y_pred = np.array(labels)
    cm = ClusteringMetric(X=data, y_pred=y_pred)

    print(cm.dunn_index())
    print(cm.DI())

labels1= label(matrix1)
print("dunnIndex matrix1: ")
dunnIndex(matrix1, labels1)

labels2= label(matrix2)
#print("label 2: ", labels2)
print("\n dunnIndex matrix2: ")
dunnIndex(matrix2, labels2)

labels3= label(matrix3)
#print("label 3: ", labels3)
print("\n dunnIndex matrix3: ")
dunnIndex(matrix3, labels3)

labels4= label(matrix4) # -> 10_7717_peerj_5665_dataYM2018_neuroblastoma -> file 1
#print("label 4: ", labels4)
print("\n dunnIndex matrix4: ")
dunnIndex(matrix4, labels4)

labels5= label(matrix5) #-> journal.pone.0175818_S1Dataset_Spain_cardiac_arrest_EDITED. -> file 2
#print("label 5: ", labels5)
print("\n dunnIndex matrix5: ")
dunnIndex(matrix5, labels5)

labels6= label(matrix6)
#print("label 6: ", labels6)
print("\n dunnIndex matrix6: ")
dunnIndex(matrix6, labels6)

labels7= label(matrix7)
#print("label 7: ", labels7)
print("\n dunnIndex matrix7: ")
dunnIndex(matrix7, labels7)

labels8= label(matrix8)
#print("label 8: ", labels8)
print("\n dunnIndex matrix8: ")
dunnIndex(matrix8, labels8)
