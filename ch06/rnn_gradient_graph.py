import numpy as np
import matplotlib.pyplot as plt

N = 2 # minibatch size
H = 3 # hidden dimension
T = 20 # length of sequence

dh = np.ones((N, H)) # initial hidden state
np.random.seed(3) # fixed random seed for reproducibility
# Wh = np.random.randn(H, H) # before
Wh = np.random.randn(H, H) * 0.5 # after

norm_list = []
for t in range(T):
    dh = np.dot(dh, Wh.T)
    norm = np.sqrt(np.sum(dh**2)) / N
    norm_list.append(norm)

# Plot graph
plt.plot(np.arange(len(norm_list)), norm_list)
plt.xticks([0, 4, 9, 14, 19], [1, 5, 10, 15, 20])
plt.xlabel('time step')
plt.ylabel('norm')
plt.show()