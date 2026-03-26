import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

img = Image.open('iisertvm-img.jpeg').convert('L')

imgMatrix = np.array(img, dtype=np.float64)
U, sigma, V_t = np.linalg.svd(imgMatrix, full_matrices=False)

print("Original Image Shape:", imgMatrix.shape)
print("U Shape:", U.shape)
print("Sigma Shape:", sigma.shape)
print("V^T Shape:", V_t.shape)

k1 = 25
k2 = 100

compressed_img_k1 = np.dot(U[:, :k1], np.dot(np.diag(sigma[:k1]), V_t[:k1, :]))
compressed_img_k2 = np.dot(U[:, :k2], np.dot(np.diag(sigma[:k2]), V_t[:k2, :]))

print("Compressed Image with k=25 Shape:", compressed_img_k1.shape)
print("Compressed Image with k=100 Shape:", compressed_img_k2.shape)

plt.figure(figsize=(10, 5))
plt.subplot(3, 1, 1)
plt.title('Original Image')
plt.imshow(imgMatrix, cmap='gray')
plt.axis('off')
plt.subplot(3, 1, 2)
plt.title('Compressed Image (k=25)')
plt.imshow(compressed_img_k1, cmap='gray')
plt.axis('off')
plt.subplot(3, 1, 3)
plt.title('Compressed Image (k=100)')
plt.imshow(compressed_img_k2, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()