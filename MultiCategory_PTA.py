import torch
import torchvision
import numpy as np
import torchvision.datasets as ds
import torchvision.transforms as transforms

mnist_train = ds.MNIST(root='./data', train=True, download=True, transform=None)
mnist_test = ds.MNIST(root='./data', train=False, download=True, transform=None)

def step_func(v):
    for elem in range(len(v)):
        if v[elem] >= 0:
            return 1
        else:
            return 0

def bin_rep(num):
    rep = np.zeros((10, 1))
    np.put(rep, num, 1)

    return rep

def mc_pta(eta, threshold, n, dataset):
    W = torch.rand(10, 784)
    #W = np.random.uniform(0, 7840, (1))
    epoch = 0
    epoch_err = []
    #print(epoch_err)
    v = 0

    transform = transforms.Compose([transforms.ToTensor()])

    x = []
    # x = np.array(x)
    #x = np.reshape(x, (784, 1))
    #print(x.shape)


    while True:
        epoch_err.extend([0])
        print(epoch)
        #print(epoch_err)
        # mcf loop:
        for i in range(n):
            #if (W * dataset[i][1]) > v:
            x = transform(dataset[i][0])
            x = np.array(x)
            x = np.reshape(x, (784, 1))

            v = np.dot(W, x)
            guess = np.argmax(v)
            
            if dataset[i][1] != guess:
                epoch_err[epoch] += 1
        
        epoch += 1
        

        # weights loop
        for j in range(n):
            desired = dataset[j][1]
            w_xi = step_func(np.dot(W, x))
            W = W + (eta * (bin_rep(desired) - w_xi)) * np.transpose(x)
        
        if (epoch_err[epoch - 1] / n) <= threshold:
            print('conv')
            break

    
    return W, epoch

def test_mc_pta(eta, threshold, n, dataset, W):
    errors = 0
    transform = transforms.Compose([transforms.ToTensor()])
    x_prime = transform(dataset[0][0])
    x_prime = np.array(x_prime)
    x_prime = np.reshape(x_prime, (784, 1))

    for i in range(10000):
        v = np.dot(W, x_prime)
        guess = np.argmax(v)

        if dataset[i][1] != guess:
                errors += 1
    
    return errors

#print(mnist_train[0][1])
#print(mnist_test[0][1])
# train_images = []
# train_images = np.array(train_images)

# train_labels = []
# train_labels = np.array(train_labels)

# test_images = []
# test_images = np.array(test_images)

# test_labels = []
# test_labels = np.array(test_labels)

# transform = transforms.Compose([transforms.ToTensor()])

# for data in range(len(mnist_train)):
#     train_images = np.append(train_images, transform(mnist_train[data][0]))
#     # train_images[data] = np.reshape(train_images, (784, 1))
#     train_labels = np.append(train_labels, bin_rep(mnist_train[data][1]))

# for data in range(len(mnist_test)):
#     test_images.append(mnist_test[data][0])
#     test_labels.append(bin_rep(mnist_test[data][1]))

# train_images= np.reshape(train_images, (784, 1))
# print(train_images[0].shape)

W, epoch = mc_pta(1, 0, 50, mnist_train)
print(epoch)
#test_mc_pta(1, 0, 10, mnist_test, W)