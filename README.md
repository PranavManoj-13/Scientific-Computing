# DSC412: Scientific Computing Algorithms

This repository contains implementations of various numerical linear algebra algorithms in Python, developed for the DSC412 course. The code demonstrates fundamental methods for solving linear systems, matrix factorizations, and optimization techniques.

## Algorithms Implemented

### Linear System Solvers
- **Conjugate Gradient Method** (`Conjugate-Grad.py`) - Iterative method for solving large sparse linear systems
- **Gaussian Elimination** (`GaussElimination.py`) - Direct method with scaled partial pivoting for solving linear systems
- **LU Factorization** (`LU-Factorization.py`) - Decomposes a matrix into lower and upper triangular matrices

### Matrix Factorizations
- **QR Factorization** (`QR-Factorization.py`) - Decomposes a matrix into orthogonal and upper triangular matrices
- **Singular Value Decomposition (SVD)** (`SVD.py`) - Factorizes a matrix into singular values and orthogonal matrices

### Optimization Methods
- **Steepest Descent** (`SteepestDescent.py`) - Gradient-based optimization algorithm
- **Steepest Descent II** (`SD-NewtonRaphson.py`) - Optimization algorithm with Newton Raphson Method to find alpha.
- **Modified Newton Method** (`ModifiedNewton.py`) - Non-linear optimization algorithm

### Applications
- **SVD Image Compression** (`SVD-Compression.py`) - Demonstrates image compression using SVD on grayscale images

## Requirements

- Python 3.x
- NumPy
- Matplotlib (for visualization)
- PIL (Pillow) for image processing in SVD-Compression.py
- Autograd for gradient and hessian computation in SD-NewtonRaphson.py and ModifiedNewton.py

Install dependencies:
```bash
pip install numpy matplotlib pillow autograd
```

## 📁 File Structure

```
DSC412-Codebase/
├── Conjugate-Grad.py      # Conjugate gradient solver
├── GaussElimination.py    # Gaussian elimination with pivoting
├── LU-Factorization.py    # LU decomposition
├── QR-Factorization.py    # QR decomposition
├── SVD.py                 # Singular value decomposition
├── SVD-Compression.py     # Image compression using SVD
├── SteepestDescent.py     # Steepest descent optimization
├── SD-NewtonRaphson.py    # SD with Newton Raphson Method
├── ModifiedNewton.py      # Modified Newton's Method
├── img/                   # Directory for images
└── README.md              # This file
```

## Implementation Notes

- All implementations use NumPy for efficient numerical computations
- Error handling is included for singular matrices and invalid inputs
- Interactive input for flexibility in testing different cases
- Verification steps included in factorization methods

## Learning Objectives

This codebase demonstrates:
- Direct and iterative methods for linear systems
- Matrix decomposition techniques
- Numerical stability considerations
- Applications of linear algebra in data compression
- Gradient-based optimization

## Contributing

This is an educational project. Feel free to:
- Report bugs or suggest improvements
- Add more algorithms or examples
- Optimize existing implementations
