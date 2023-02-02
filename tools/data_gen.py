import numpy as np

if __name__ == '__main__':
    n = 1024 * 32
    dim = 4
    data = np.random.uniform(0.0, 100.0, (n, dim))
    data.astype('float32').tofile('../data/q_4f.dat')

    print(data)
