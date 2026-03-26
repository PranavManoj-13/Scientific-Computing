import numpy as np

n = int(input("Enter the size of the square matrix: "))
A = np.zeros((n, n))

print("Enter the elements of the matrix:")
for i in range(n):
    for j in range(n):
        A[i][j] = int(input(f" A [{i+1}][{j+1}] = "))

ATA = A.T @ A
eigenvalues, eigenvectors = np.linalg.eig(ATA)
singular_values = np.sqrt(np.abs(eigenvalues))

idx = np.argsort(-singular_values)
singular_values = singular_values[idx]
eigenvectors = eigenvectors[:, idx]

U = np.zeros((n, n))

for i in range(n):
    if singular_values[i] > 1e-10:
        U[:, i] = (A @ eigenvectors[:, i]) / singular_values[i]

sigma = np.zeros((n, n))

for i in range(n):
    sigma[i, i] = singular_values[i]

V_t = eigenvectors.T

print("Matrix U:")
print(U)
print("Matrix Sigma:")
print(sigma)
print("Matrix V^T:")
print(V_t)