import numpy as np

shape = (64, 64)

shapes = []

shapes.append(10)

for value in shape:
    shapes.append(value)

a = np.zeros(shape=shapes)

print(a.shape)
