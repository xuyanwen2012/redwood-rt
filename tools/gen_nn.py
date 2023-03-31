import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_points", type=int, default=1048576,
                    help="Number of points to generate (default: 1048576)")
parser.add_argument("--filename", type=str, default="../data/input_nn_1m_4f.dat",
                    help="Name of the output file (default: ../data/input_nn_1m_4f.dat)")
parser.add_argument("--distribution", type=str, default="uniform",
                    help="Type of distribution to use (default: uniform)")
args = parser.parse_args()

# Define the dimensions of the array
dimensions = 4

# Define the range of the coordinates
low = 0.0
high = 1024.0

# Generate the random points as float32
if args.distribution == "uniform":
    points = np.random.uniform(low, high, size=(
        args.num_points, dimensions)).astype(np.float32)
elif args.distribution == "normal":
    mean = (high + low) / 2.0
    std_dev = (high - low) / 6.0
    points = np.random.normal(mean, std_dev, size=(
        args.num_points, dimensions)).astype(np.float32)
else:
    raise ValueError("Invalid distribution type")

# Write the points to a binary file
with open(args.filename, "wb") as f:
    f.write(points.tobytes())

# Print 20 random points for inspection
print("Randomly selected points for inspection:")
for point in np.random.choice(points, 20):
    print(point)
