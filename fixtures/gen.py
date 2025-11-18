import numpy as np

arr = np.random.randn(64 * 64 * 4).astype('float64')
arr.tofile("test_array.bin")