import numpy as np

# Define the dimensions of the array
num_points = 2 * 1024 * 1024
dimensions = 4

# Generate the random points as float32
points = np.random.uniform(low=0.0, high=1024.0, size=(
    num_points, dimensions-1)).astype(np.float32)
w_values = np.ones((num_points, 1), dtype=np.float32)
points = np.hstack((points, w_values))

# Write the points to a binary file
with open("../data/input_bh_2m_4f.dat", "wb") as f:
    f.write(points.tobytes())
