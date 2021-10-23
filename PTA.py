import numpy as np
from matplotlib import pyplot as plt

def step_func(v):
    if v >= 0:
        return 1
    else:
        return 0

def pta(eta, S, S_star, S1, S0):

    W_prime = np.random.uniform(-1, 1, (3, 1)) #omega Ω
    S1_prime = S[(np.dot(S_star, W_prime) >= 0).ravel()]
    S0_prime = S[(np.dot(S_star, W_prime) < 0).ravel()]

    mcf = 1
    mcf_num = []
    epoch = 0

    while mcf != 0:
        mcf = 0
        epoch += 1
        S1_prime = S[(np.dot(S_star, W_prime) >= 0).ravel()]
        S0_prime = S[(np.dot(S_star, W_prime) < 0).ravel()]
        for x in range(len(S)):
            if (S[x] in S1) and (S[x] not in S1_prime):
                W_prime[0] = W_prime[0] + (eta * S_star[x][0])
                W_prime[1] = W_prime[1] + (eta * S_star[x][1])
                W_prime[2] = W_prime[2] + (eta * S_star[x][2])
                mcf += 1
            
            elif (S[x] in S0) and (S[x] not in S0_prime):
                W_prime[0] = W_prime[0] - (eta * S_star[x][0])
                W_prime[1] = W_prime[1] - (eta * S_star[x][1])
                W_prime[2] = W_prime[2] - (eta * S_star[x][2])
                mcf += 1
        mcf_num.append(mcf)

    return epoch, mcf_num, W_prime


# Pick w0 uniformly at random on [-1/4, 1/4]
w0 = np.random.uniform(-.25, .25, (1, 1))


# Pick w1, w2 uniformly at random on [-1, 1] 
w12 = np.random.uniform(-1, 1, (2, 1))

W = np.concatenate((w0, w12), axis=0)

# Pick n = 100 vectors x1...xn independently and uniformly
# at random on [-1, 1]^2, call the collection S 
S = np.random.uniform(-1, 1, (100, 2))

# Let S1 ⊂ S denote the collection of all
# x = [x1 x2] ∈ S satisfying [1 x1 x2][w0 w1 w2]^T >= 0
S_star = np.concatenate((np.ones((100, 1)), S), axis=1)

S1 = S[(np.dot(S_star, W) >= 0).ravel()]

# Let S0 ⊂ S denote the collection of all
# x = [x1 x2] ∈ S satisfying [1 x1 x2][w0 w1 w2]^T < 0
S0 = S[(np.dot(S_star, W) < 0).ravel()]

# In one plot, show the line w0 + w1x1 + w2x2 = 0, with x1
# being the "x-axis" and x2 being the "y-axis". In the same
# plot show all the points in S1 and all the points in S0.
# Use different symbols for S0 & S1. Indicate which points
# belong to which class.
fig, axis = plt.subplots()
axis.scatter(S1[:, 0], S1[:, 1], label="S1 pts")
axis.scatter(S0[:, 0], S0[:, 1], label="S0 pts")

x1 = np.linspace(-1, 1, 100)
x2 = -(W[0] + W[1]*x1) / W[2]
axis.plot(x1, x2, 'g', label="Boundary")

axis.legend(loc="lower right")
plt.title("A plot of Sl & S0 points, seperated by the line \nw0 + w1x1 + w2x2 = 0")
plt.show()

epoch_1, mcf_1, weights_1 = pta(1, S, S_star, S1, S0)
eplist_1 = []
for x in range(epoch_1):
    eplist_1.append(x)

epoch_10, mcf_10, weights_10 = pta(10, S, S_star, S1, S0)
eplist_10 = []
for x in range(epoch_10):
    eplist_10.append(x)

epoch_pt1, mcf_pt1, weights_pt1 = pta(.1, S, S_star, S1, S0)
eplist_pt1 = []
for x in range(epoch_pt1):
    eplist_pt1.append(x)

fig, axis1 = plt.subplots()
axis1.plot(eplist_1, mcf_1, 'y', label="η = 1")
axis1.plot(eplist_10, mcf_10, 'g', label="η = 10")
axis1.plot(eplist_pt1, mcf_pt1, 'r', label="η = .1")

axis1.legend(loc="upper right")
plt.title("Epoch Number vs Number of Misclassifications,\nfor η = 0.1, 1, 10")
plt.show()