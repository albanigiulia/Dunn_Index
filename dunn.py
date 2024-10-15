import random
from random import randint

#creo una matrice di 0 e 1
matrix_zero_uno = []

for i in range(10):
    n = []
    for j in range(10):
        number = randint(0,1)
        n.insert(i,number)
    matrix_zero_uno.append(n)
print(matrix_zero_uno)

#creo una matrice di numeri compresi fra 0 e 1
matrix_compresi = []

for i in range(10):
    n = []
    for j in range(10):
        number = random.uniform(0, 1)
        n.insert(i,number)
    matrix_compresi.append(n)
print(matrix_compresi)

#creo una matrice di numeri compresi fra 0 e 9
matrix3 = []

for i in range(10):
    n = []
    for j in range(10):
        number = random.uniform(0, 9)
        n.insert(i,number)
    matrix_compresi.append(n)
print(matrix_compresi)


