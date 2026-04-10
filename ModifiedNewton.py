import autograd.numpy as np
from autograd import grad, jacobian
import matplotlib.pyplot as plt

def newtonLineSearch(grad_f, hessian_f, xk, pk, tol=1e-6, max_iter=50):
    alpha = 1.0

    for _ in range(max_iter):
        x_new = xk + alpha * pk
        g = grad_f(x_new)
        H = hessian_f(x_new)

        phi_prime = np.dot(g, pk)
        phi_double_prime = np.dot(pk, np.dot(H, pk))

        if abs(phi_double_prime) < 1e-10:
            break
        alpha_new = alpha - phi_prime / phi_double_prime

        if abs(alpha_new - alpha) < tol:
            return alpha_new
        alpha = alpha_new

    return alpha

def modifiedNewtonMethod(f, x0, tol=1e-6, max_iter=100):
    xk = np.array(x0)
    grad_f = grad(f)
    hessian_f = jacobian(grad_f)
    path = [xk.copy()]

    for k in range(max_iter):
        g = grad_f(xk)
        H = hessian_f(xk)

        if np.linalg.norm(g) < tol:
            break

        pk = -np.linalg.solve(H, g)
        alpha_k = newtonLineSearch(grad_f, hessian_f, xk, pk)
        xk = xk + alpha_k * pk
        
        print(f"Iter {k}: alpha = {alpha_k:.6f}, x = {xk}")
        
        path.append(xk.copy())

    return np.array(path)

def f(x):
    return x[0]**2 + 3*x[1]**2 + x[0]*x[1]

x0 = np.array([4.0, -4.0])
path = modifiedNewtonMethod(f, x0)

x_vals = np.linspace(-6, 6, 100)
y_vals = np.linspace(-6, 6, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = X**2 + 3*Y**2 + X*Y

plt.figure()
plt.contour(X, Y, Z, levels=30)
plt.plot(path[:, 0], path[:, 1], 'ro-')

for i in range(len(path)-1):
    dx = path[i+1][0] - path[i][0]
    dy = path[i+1][1] - path[i][1]
    plt.arrow(path[i][0], path[i][1], dx, dy,
              head_width=0.2, length_includes_head=True)

plt.show()