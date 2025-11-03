import numpy as np

inane = np.zeros((11, 9))
stupid = np.arange(9)
rows = 11

crazy = np.arange(9) - 4

worst = np.full((np.shape(inane)), stupid)
print(worst * crazy)


crab = np.arange(100)
lobster = crab.reshape((10, 10))
print(lobster)

cram = np.arange(19) * 5
print(cram)
