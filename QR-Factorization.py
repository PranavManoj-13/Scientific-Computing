import numpy as np

m = int(input("Enter number of rows: "))
n = int(input("Enter number of columns: "))

A = np.zeros((m, n))

print("Enter the elements of the matrix:")
for i in range(m):
    for j in range(n):
        A[i][j] = float(input(f" A [{i+1}][{j+1}] = "))

mode = input("Choose QR method (type 'reduced' or 'full'): ").strip().lower()

Q = np.zeros((m, n))
R = np.zeros((n, n))

for j in range(n):
    v = A[:, j].copy()
    
    for k in range(j):
        R[k, j] = np.dot(Q[:, k], A[:, j])
        v -= R[k, j] * Q[:, k]
    
    R[j, j] = np.linalg.norm(v)
    
    if R[j, j] < 1e-10:
        raise ValueError("Matrix A has linearly dependent columns.")
    
    Q[:, j] = v / R[j, j]

if mode == 'reduced':
    Q_final = Q
    R_final = R

elif mode == 'full':
    Q_full = np.zeros((m, m))
    Q_full[:, :n] = Q

    for j in range(n,m):
        v = np.random.rand(m)

        for i in range(j):
            v -= np.dot(Q_full[:, i], v) * Q_full[:, i]

            v = v / np.linalg.norm(v)
            Q_full[:, j] = v
    
    R_full = np.zeros((m, n))
    R_full[:n, :] = R

    Q_final = Q_full
    R_final = R_full

else:
    raise ValueError("Invalid mode selected. Please choose 'reduced' or 'full'.")

print("Matrix Q:")
print(Q_final)
print("Matrix R:")
print(R_final)

A_reconstructed = Q_final @ R_final
print("Reconstructed A (Q @ R):")
print(A_reconstructed)

print("Q^T @ Q:")
print(Q_final.T @ Q_final)

error = np.linalg.norm(A - A_reconstructed)
print(f"Reconstruction error (||A - Q @ R||): {error:.2e}")