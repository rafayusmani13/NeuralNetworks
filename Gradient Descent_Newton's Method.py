import numpy as np
import matplotlib.pyplot as plt

def f_xy(w):
    ans = -np.log(1 - w[0] - w[1]) - np.log(w[0]) - np.log(w[1])
    return ans

def gradient_func(w):
    gradient = np.array([-1/(1 - w[0] - w[1]) - 1/w[0], -1/(1 - w[0] - w[1]) - 1/w[1]])
    return gradient

def hessian_func(w):
    total_hessian = np.array([[1/(1 - w[0] - w[1])**2 + 1/w[0]**2, 1/(1 - w[0] - w[1])**2],
                            [1/(1 - w[0] - w[1])**2, 1/(1 - w[0] - w[1])**2 + 1/w[1]**2]])
    return total_hessian
                  
def newton_method(w, eta, totalIterations, precision):
    prev_step = 1 #error rate
    #prev_w = w
    iterations = 0
    iter_list = [1]
    point_list = [w]
    err = 0
    cost_list = [] #errors
    

    while prev_step > precision and iterations < totalIterations:
        err = f_xy(w)
        cost_list.append(err)
        inverse = np.linalg.inv(hessian_func(w))
        w = w - (eta * np.matmul(inverse, gradient_func(w)))
        temp_err = f_xy(w)
        cost_list.append(temp_err)
        prev_step = abs(temp_err - err)
        iterations += 1
        iter_list.append(iterations)
        point_list.append(w)
        #print("Iteration", iterations,"\nX value is", w[0], "\nY value is", w[1]) #Print iterations
    
    #print("The local minimum occurs at", w[0], ", ", w[1])
    return point_list, iter_list, cost_list


def gradient_descent(w, eta, totalIterations, precision):
    # precision = threshold
    prev_step = 1 #error rate
    #prev_w = w
    iterations = 0
    iter_list = []
    point_list = [w]
    err = f_xy(w)
    cost_list = [] #errors
    cost_list.append(err)
    

    while prev_step > precision and iterations < totalIterations:
        iter_list.append(iterations)
        err = f_xy(w)
        cost_list.append(err)
        w = w - eta * gradient_func(w)
        #print('W: ', w, '\n\n\n\n', f_xy(w))
        temp_err = f_xy(w)
        cost_list.append(temp_err)
        prev_step = abs(temp_err - err)
        iterations += 1
        point_list.append(w)
        #print("Iteration", iterations,"\nX value is", w[0], "\nY value is", w[1]) #Print iterations
    
    #print("The local minimum occurs at", w[0], ", ", w[1])
    #print(point_list.shape, iter_list.shape, cost_list.shape)
    # print(temp_err)
    return point_list, iter_list, cost_list

# w ← w − η∇E(w)

inDomain = False
while inDomain == False:
    w0 = np.random.rand(2, 1)
    if ((w0[0] + w0[1]) < 1):
        inDomain = True

points, epochs, errors = gradient_descent(w0, 1, 100, .001)

#fig, axis1 = plt.subplot()
epochs = np.array(epochs)
errors = np.array(errors)
print(epochs)
# plt.scatter(points, errors)
# plt.show()