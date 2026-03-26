import numpy as np

max_iter = 1000
tol = 1e-8

n = int(input("Enter number of dimensions: "))

print("Enter the elements of the matrix A:")
A = np.zeros((n, n))
for i in range(n):   
    for j in range(n):
        A[i][j] = int(input(f"A[{i}][{j}]: "))

print("Enter the elements of the vector b:")
b = np.zeros(n)
for i in range(n):
    b[i] = int(input(f"b[{i}]: "))

print("Enter the initial guess for the solution vector x:")
x = np.zeros(n)
for i in range(n):
    x[i] = int(input(f"x[{i}]: "))

r = b - A @ x
p = r.copy()

for i in range(max_iter):
    Ap = A @ p
    alpha = (r.T @ r) / (p.T @ Ap)
    x_new = x + alpha * p
    r_new = r - alpha * Ap

    print(f"Iteration {i+1}: x = {x_new}, Residual = {np.linalg.norm(r_new)}")

    if np.linalg.norm(r_new) < tol:
        break

    beta = (r_new.T @ r_new) / (r.T @ r)
    p = r_new + beta * p
    r = r_new
    x = x_new

print("Approx. Solution:", x)
print("Final Residual:", np.linalg.norm(r))