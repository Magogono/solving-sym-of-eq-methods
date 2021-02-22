import numpy as np
import math
import solve
import time
import matplotlib.pyplot as plt

# parametry: f=5, e=9, c=1, d=2


def create_A_b(a1, a2, a3, N):
    # tworzenie macierzy A(NxN)
    A = np.zeros((N, N))

    for i in range(0, N):
        for j in range(0, N):
            if i == j:
                A[i][j] = a1
            elif i == j + 1 or i == j - 1:
                A[i][j] = a2
            elif i == j + 2 or i == j - 2:
                A[i][j] = a3

    # tworzenie wektora b(N)
    b = np.zeros(N)
    for n in range(0, len(b)):
        b[n] = math.sin(n*6)

    return A, b


A_mat, b_vect = create_A_b(5.0+9.0, -1.0, -1.0, 9*1*2)


# ========= metody iteracyjne ===========
print("Zadanie B")

start = time.time()
_, it_jacobi = solve.jacobi(A_mat, b_vect)
end = time.time()
print(" Metoda Jacobiego: " + str(it_jacobi) + " iteracji, czas: " + str((end-start)*1000) + " ms")

start = time.time()
_, it_gauss_seidel = solve.gauss_seidel(A_mat, b_vect)
end = time.time()
print(" Metoda Gaussa-Seidla: " + str(it_gauss_seidel)+ " iteracji, czas: " + str((end-start)*1000) + " ms\n")


print("Zadanie C")

# tworzenie nowego układu równań
A_mat, b_vect = create_A_b(3.0, -1.0, -1.0, 9*1*2)


# obliczenia
start = time.time()
_, it_jacobi = solve.jacobi(A_mat, b_vect)
end = time.time()
print(" Metoda Jacobiego: " + str(it_jacobi) + " iteracji, czas: " + str((end-start)*1000) + " ms")

start = time.time()
_, it_gauss_seidel = solve.gauss_seidel(A_mat, b_vect)
end = time.time()
print(" Metoda Gaussa-Seidla: " + str(it_gauss_seidel)+ " iteracji, czas: " + str((end-start)*1000) + " ms\n")


print("Zadanie D")
# faktoryzacja LU
x = solve.factorization_LU(A_mat, b_vect)

res = A_mat @ x - b_vect

print(" Norma z residuum: " + str(solve.norm_2(res)) + "\n")


N = [100, 500, 1000, 2000, 3000]
time_jacobi = []
time_gauss_seidel = []
time_factorization_LU = []


# jacobi
for size in N:
    # tworzenie macierzy i wektora wyrazów wolnych
    A_mat, b_vect = create_A_b(5.0 * 9.0, -1.0, -1.0, size)

    # mierzenie czasu rozwiązywania układu metodą jacobiego
    start = time.time()
    solve.jacobi(A_mat, b_vect)
    end = time.time()

    # zapisanie wyniku w tablicy czasów
    time_jacobi.append(end-start)

plt.plot(N, time_jacobi)
plt.title('Metoda Jacobiego')
plt.xlabel('rozmiar macierzy N')
plt.ylabel('czas [s]')
plt.show()

# gauss-seidel
for size in N:
    # tworzenie macierzy i wektora wyrazów wolnych
    A_mat, b_vect = create_A_b(5.0 * 9.0, -1.0, -1.0, size)

    # mierzenie czasu rozwiązywania układu metodą Gaussa-Seidla
    start = time.time()
    solve.gauss_seidel(A_mat, b_vect)
    end = time.time()

    # zapisanie wyniku w tablicy czasów
    time_gauss_seidel.append(end - start)

plt.plot(N, time_gauss_seidel)
plt.title('Metoda Gaussa-Seidla')
plt.xlabel('rozmiar macierzy N')
plt.ylabel('czas [s]')
plt.show()

N = [100, 250, 500, 1000, 1500]

# faktoryzacja LU
for size in N:
    # tworzenie macierzy i wektora wyrazów wolnych
    A_mat, b_vect = create_A_b(5.0 * 9.0, -1.0, -1.0, size)

    # mierzenie czasu rozwiązywania układu metodą faktoryzacji LU
    start = time.time()
    solve.factorization_LU(A_mat, b_vect)
    end = time.time()

    # zapisanie wyniku w tablicy czasów
    time_factorization_LU.append(end - start)

plt.plot(N, time_factorization_LU)
plt.title('Faktoryzacja LU')
plt.xlabel('rozmiar macierzy N')
plt.ylabel('czas [s]')
plt.show()

