import numpy as np
import matplotlib.pyplot as plt
test = np.genfromtxt("n14_i100_mpi.csv", delimiter=",")
my = np.genfromtxt("rajev.csv", delimiter=",")
my = my.round(6)
result = my == test
print(result.all())
# print(test[result])
# print(my[result])

plt.imshow(my, cmap='hot', interpolation='nearest', vmin=0, vmax=100)
plt.imshow(test, cmap='hot', interpolation='nearest', vmin=0, vmax=100)