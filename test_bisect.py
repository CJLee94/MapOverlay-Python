import bisect
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Test sort
array_lengths = np.logspace(1, 4, 50)
test_result = np.zeros((20, len(array_lengths)))

for jj, length in enumerate(array_lengths):
    for ii in range(20):
        test = np.random.randint(-100, 100, int(length))
        sorted_list = []
        tic = time.time()
        for elem in test:
            bisect.insort(sorted_list, elem)
        toc = time.time()
        test_result[ii, jj] = toc - tic

x = np.stack([array_lengths]*test_result.shape[0], axis=0)

plt.scatter(x.flatten(), test_result.flatten(), label="Test Points")
f = interp1d(x[0], test_result.mean(axis=0), kind="cubic")
xnew = np.logspace(1, 4, 300)
plt.plot(xnew, f(xnew), color="orange", label="Mean for each length")
plt.legend()
plt.title("Bisect Performance on Sorting")
plt.xlabel("Input Length")
plt.ylabel("Time")
# plt.show()

array_lengths = np.logspace(1, 4, 50)
test_result = np.zeros((100, len(array_lengths)))

for jj, length in enumerate(array_lengths):
    for ii in range(20):
        test = np.random.randint(-100, 100, int(length))
        sorted_list = []
        for elem in test:
            if elem == test[-1]:
                tic = time.time()
                bisect.insort(sorted_list, elem)
                toc = time.time()
            else:
                bisect.insort(sorted_list, elem)
        # toc = time.time()
        test_result[ii, jj] = toc - tic

x = np.stack([array_lengths]*test_result.shape[0], axis=0)

plt.figure()
plt.scatter(x.flatten(), test_result.flatten(), label="Test Points")
f = interp1d(x[0], test_result.mean(axis=0), kind="cubic")
xnew = np.logspace(1, 4, 300)
plt.plot(xnew, f(xnew), color="orange", label="Mean for each length")
plt.legend()
plt.title("Bisect Performance on Insertion")
plt.xlabel("Input Length")
plt.ylabel("Time")
plt.show()

