import numpy as np
from matplotlib import pyplot as plt

# The output neuron will use the activation function φ(v) = v
def output_func(v):
    return v

# all other neurons will use the activation function φ(v) = tanh(v)
def neuron_func(v):
    return np.tanh(v)

def back_neuron_func(v):
    return 1 - (np.tanh(v)**2)
    

def predict(inputs, w1, b1, w2, b2):
    ans = []

    for ind in range(len(inputs)):
        x = np.concatenate((ones, x_vals[ind].reshape(1,1)))
        sum_1 = np.matmul(w1, x)
        hidden = neuron_func(sum_1)
        
        _hidden = np.concatenate((ones, hidden))
        sum_2 = np.matmul(w2, _hidden)
        output = output_func(sum_2) 
        ans.append(output)
    
    return ans

# Draw n = 300 real numbers uniformly at random on [0, 1], call them x1, . . . , xn.
x_vals = np.random.uniform(0, 1, (300, 1))

# Draw n real numbers uniformly at random on [−1/10, 1/10], call them ν1, . . . , νn.
v_vals = np.random.uniform(-0.1, 0.1, (300, 1))

# Let di = sin(20xi) + 3xi + νi, i = 1, . . . , n. Plot the points (xi, di), i = 1, . . . , n
d = np.sin(20*x_vals) + 3 * x_vals + v_vals

ones = np.ones((1,1))
np.random.normal(1, 1, 1)

# weights
_w1 = np.random.normal(1, 1, (24, 1))
b1 = np.random.normal(0, 1, (24, 1))
_w2 = np.random.normal(1, 1, (1, 24))
b2 = np.random.normal(0, 1, (1, 1))

w1 = np.concatenate((b1, _w1), axis=1)
w2 = np.concatenate((b2, _w2), axis=1)


eta = 0.01
epoch = 0
error = np.inf
error_list = []

while epoch < 10000:
    output_list = []
    for ind in range(300):
        x = np.concatenate((ones, x_vals[ind].reshape(1,1)))
        sum_1 = np.matmul(w1, x)
        hidden = neuron_func(sum_1)
        
        _hidden = np.concatenate((ones, hidden))
        sum_2 = np.matmul(w2, _hidden)
        output = output_func(sum_2)
        output_list.append(output[0][0])

        delta_2 = d[ind] - output
        delta_1 = np.matmul(w2.T, delta_2)[1:, :] * back_neuron_func(sum_1)

        w1 += eta * np.matmul(delta_1, x_vals[ind:ind+1])
        w2 += eta * np.matmul(delta_2, _hidden.T)
    
        
    
    error = sum((d[i][0]-output_list[i])**2 for i in range(300))/300
    error_list.append(error)


    # if error_list[epoch] > error_list[epoch - 1] and epoch > 0:
    #     eta *= .9

    if error_list[epoch] <= 0.001:
      break
    
    epoch += 1
    error = 0

# Plot the points (xi, di), i = 1, . . . , n
fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(x_vals, d)
plt.title('Plot of Input')
plt.xlabel('X')
plt.ylabel('D')

ans = predict(x_vals, w1, b1, w2, b2)
fig, ax1 = plt.subplots()
ax1.scatter(x_vals, d, color='b')
ax1.scatter(x_vals, ans, color='r')
plt.title('Plot of Input v Output')
plt.xlabel('Input')
plt.ylabel('Output')
plt.show()


fig, ax2 = plt.subplots()
ax2.plot(error_list, color='b')
plt.title('Plot of MSE v Epochs')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.show()