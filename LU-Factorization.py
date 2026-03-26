import numpy as np

def checkLU(A):
    rows, cols = A.shape

    if rows != cols:
        raise ValueError("Matrix A must be square.")
    
    for i in range(rows):
        minor = A[:i+1, :i+1]
        if abs(np.linalg.det(minor)) < 1e-10:
            raise ValueError("Matrix A is not suitable for LU factorization.")
    
    print("Matrix A is suitable for LU factorization.")
    return True

def LU(A):
    if not checkLU(A):
        return None, None
    
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()

    for k in range(n-1):
        for i in range(k+1, n):
            L[i, k] = U[i, k] / U[k, k]

            for j in range(k, n):
                U[i, j] = U[i, j] - L[i, k] * U[k, j]
    
    return L, U

n = int(input("Enter number of dimensions: "))
A = np.zeros((n, n))

print("Enter the elements of the matrix:")
for i in range (n):
    for j in range (n):
        A[i][j] = float(input(f" A [{i+1}][{j+1}] = "))

L, U = LU(A)
if L is not None and U is not None:
    print("L:")
    print(L)
    print("U:")
    print(U)

    print("Verification (L @ U):")
    print(L @ U)