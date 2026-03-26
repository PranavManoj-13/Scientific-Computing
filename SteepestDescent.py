import numpy as np
import matplotlib.pyplot as plt

n = int(input("Enter number of dimensions: "))

A = np.zeros((n, n))
b = np.zeros(n)
x = np.zeros(n)

print("Enter the elements of the matrix A:")
for i in range(n):
    for j in range(n):
        A[i][j] = int(input(f"A[{i}][{j}]: "))

print("Enter the elements of the vector b:")
for i in range(n):
    b[i] = int(input(f"b[{i}]: "))

print("Enter the initial guess for the solution vector x:")
for i in range(n):
    x[i] = int(input(f"x[{i}]: "))

def f(x):
    return 0.5 * x.T @ A @ x - b.T @ x

def grad_f(x):
    return A @ x - b

def steepestDescent(x0, max_iter=1000, tol=1e-8):
    x = x0.copy()

    path = [x.copy()]
    losses = [f(x)]

    for i in range(max_iter):
        g = grad_f(x)

        if np.linalg.norm(g) < tol:
            break

        alpha = (g.T @ g) / (g.T @ A @ g)
        x = x - alpha * g

        path.append(x.copy())
        losses.append(f(x))
    
    return np.array(path), np.array(losses)

path, losses = steepestDescent(x)
print("Approx. Solution:", path[-1])

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(losses, marker='o')
ax[0].set_title("Loss vs Iteration")
ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("Loss")
ax[0].grid()

if n == 2:
    x0_vals = np.linspace(min(path[:, 0].min(), -5), max(path[:, 0].max(), 5), 200)
    x1_vals = np.linspace(min(path[:, 1].min(), -5), max(path[:, 1].max(), 5), 200)
    X0, X1 = np.meshgrid(x0_vals, x1_vals)
    Z = 0.5 * (A[0,0]*X0**2 + 2*A[0,1]*X0*X1 + A[1,1]*X1**2) - b[0]*X0 - b[1]*X1

    ax[1].contour(X0, X1, Z, levels=30, cmap='viridis')
    ax[1].plot(path[:, 0], path[:, 1], marker='o', color='red', lw=2)
    ax[1].set_title("Contour + Path of Steepest Descent")
    ax[1].set_xlabel("x[0]")
    ax[1].set_ylabel("x[1]")
    ax[1].grid()
else:
    ax[1].plot(path[:, 0], path[:, 1], marker='o')
    ax[1].set_title("Path of Steepest Descent (first 2 dims)")
    ax[1].set_xlabel("x[0]")
    ax[1].set_ylabel("x[1]")
    ax[1].grid()

plt.tight_layout()
plt.show()