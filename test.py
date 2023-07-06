import numpy as np

a = 2.14

np.save('a.npy', a)

b = np.load('a.npy')

print(b)