import numpy as np

low = 0.0
high = 1024.0
size = 1024*1024
dims = 4

kronecker = np.random.normal(loc=0, scale=1, size=(dims, size))
particles = np.exp(kronecker)
particles = (high - low) * particles / np.max(particles) + low

filename = "../data/input_1m_4f_kronecker.dat"
with open(filename, "wb") as f:
    f.write(particles.astype(np.float32).tobytes())
