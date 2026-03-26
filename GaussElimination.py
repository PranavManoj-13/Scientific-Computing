import numpy as np

def scaledPartialPivot(A, b):
    n = len(b)
    s = np.max(np.abs(A), axis=1)

    if np.any(s == 0):
        raise ValueError("Zero row detected in matrix A.")
    
    for k in range(n-1):
        ratios = np.abs(A[k:n, k]) / s[k:n]
        pivotIndex = np.argmax(ratios) + k
        
        if A[pivotIndex, k] == 0:
            raise ValueError("Matrix is singular.")
        
        if pivotIndex != k:
            A[[k, pivotIndex]] = A[[pivotIndex, k]]
            b[[k, pivotIndex]] = b[[pivotIndex, k]]
            s[[k, pivotIndex]] = s[[pivotIndex, k]]
        
        for i in range(k+1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

        if A[n-1, n-1] == 0:
            raise ValueError("Matrix is singular.")
        
        return A, b

def backSubstitution(A, b):
    n = len(b)
    x = np.zeros(n)

    for i in range(n-1, -1, -1):
        if A[i, i] == 0:
            raise ValueError("Matrix is singular.")
        
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]

    return x

def gaussElimination():
    n = int(input("Enter the number of variables: "))
    A = np.zeros((n, n))
    b = np.zeros(n)

    print("Enter the coefficients of the matrix A:")
    for i in range(n):
        for j in range(n):
            A[i, j] = float(input(f"A[{i+1}][{j+1}]: "))
    
    print("Enter the constants of the vector b:")
    for i in range(n):
        b[i] = float(input(f"b[{i+1}]: "))
    
    A, b = scaledPartialPivot(A, b)
    x = backSubstitution(A, b)

    print("The solution is:")
    for i in range(n):
        print(f"x[{i+1}] = {x[i]}")

if __name__ == "__main__":
    gaussElimination()