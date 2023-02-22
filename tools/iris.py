import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load Fisher's Iris dataset
iris = load_iris()

# Perform PCA to reduce the dimensionality to 3D
pca = PCA(n_components=3)
data_3d = pca.fit_transform(iris.data)

# Plot the data in 3D space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=iris.target)

# Add labels and a title to the plot
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('Fisher\'s Iris dataset in 3D space')

plt.show()
