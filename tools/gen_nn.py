import numpy as np

# Define the dimensions of the array
num_points = 1024 * 1024
dimensions = 4

# Define the range of the coordinates
low = 0.0
high = 1024.0

# Generate the random points as float32
points = np.random.uniform(low, high, size=(
    num_points, dimensions)).astype(np.float32)

# Write the points to a binary file
with open("../examples/data/input_nn_1m_4f.dat", "wb") as f:
    f.write(points.tobytes())
