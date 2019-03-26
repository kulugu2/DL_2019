import numpy as np
import matplotlib.pyplot as plt
def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)

def show_result(x, y, pred_y):
    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] <= 0.5:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.show()
    return

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def derivative_sigmoid(x):
    #sx = sigmoid(x)
    return np.multiply(x, 1.0-x)

def init_network():
    
    #w1 = np.random.random([2, 5]) - 0.5
    w1 = np.random.randn(2, 5) * np.sqrt(2/7)
    b1 = np.random.random([5, 1]) 
    #w2 = np.random.random([5, 5]) - 0.5
    w2 = np.random.randn(5, 5) * np.sqrt(2/10)
    b2 = np.random.random([5, 1])
    #wo = np.random.random([5, 1]) - 0.5
    wo = np.random.randn(5, 1) * np.sqrt(2/6)
    bo = np.random.random([1, 1])

    network = {"w1": w1, "b1": b1, "w2":w2, "b2":b2, "wo":wo, "bo":bo}

    return network

def forward(x, model):
    xw1 = np.add(np.dot(x, model["w1"]), model["b1"].T)
    z1 = sigmoid(xw1)
    
    xw2 = np.add(np.dot(z1, model["w2"]), model["b2"].T)
    z2 = sigmoid(xw2)

    xo = np.add(np.dot(z2, model["wo"]), model["bo"].T)
    zo = sigmoid(xo)
    #print(zo)
    
    cache = {"x": x, "xw1": xw1, "z1": z1, "xw2": xw2, "z2":z2, "xo":xo, "zo":zo}
    return zo, cache

def cost(pred, y):
    m = y.shape[0]
    j = 0.0
    for i in range(m):
        if y[i] == 1:
            if pred[i] < 0.0001:
                j += np.log(0.0001)
            else:
                j += np.log(pred[i])
        else:
            if 1.0-pred[i] < 0.0001:
                j += np.log(0.0001)
            else:
                j += np.log(1.0 - pred[i])
    #j = np.multiply(y, np.log(pred)) + np.multiply((1-y), np.log(1.0 - pred))
    
    #j = (-1 / m) * np.sum(j)
    j = (-1 / m) * j
    return j

def back_prop(model, cache, y):
    m = y.shape[0]
    #errors = y - cache["zo"]
    #errors = np.zeros((m, 1))
    #for i in range(m):
    #    if y[i][0] == 1:
    #        errors[i][0] += -y[i][0] / cache["zo"][i][0]
    #    else:
    #        errors[i][0] += (1-y[i][0]) / (1.0 - cache["zo"][i][0])
    errors = -(np.divide(y, cache["zo"]) - np.divide(1-y, 1.0-cache["zo"]))
    theta_o = np.multiply(derivative_sigmoid(cache["zo"]) , errors)
    dwo = np.dot(cache["z2"].T, theta_o) / m
    dbo = np.sum(theta_o) / m
    
    theta_2 = np.multiply(np.dot(theta_o, model["wo"].T), derivative_sigmoid(cache["z2"]))
    dw2 = np.dot(cache["z1"].T, theta_2) / m
    db2 = np.sum(theta_2) / m
    
    theta_1 = np.multiply(np.dot(theta_2, model["w2"].T), derivative_sigmoid(cache["z1"]))
    dw1 = np.dot(cache["x"].T, theta_1) / m
    db1 = np.sum(theta_1) / m
    grad = {"dwo": dwo, "dbo": dbo, "dw2":dw2, "db2":db2, "dw1":dw1, "db1":db1}
    
    return grad

def update_weight(model, grad, learning_rate):
    model["w1"] = model["w1"] - learning_rate * grad["dw1"]
    model["b1"] = model["b1"] - learning_rate * grad["db1"]
    model["w2"] = model["w2"] - learning_rate * grad["dw2"]
    model["b2"] = model["b2"] - learning_rate * grad["db2"]
    model["wo"] = model["wo"] - learning_rate * grad["dwo"]
    model["bo"] = model["bo"] - learning_rate * grad["dbo"]
    return model

def compute_acc(pred, y):
    m = y.shape[0]
    tmp = 0
    for i in range(m):
        if (pred[i] < 0.3 and y[i] == 0) or (pred[i] >= 0.3 and y[i] == 1):
            #print(pred[i], y[i])
            tmp = tmp+1
    #print(tmp, m)
    return tmp / m
if __name__ == '__main__':
    x, y = generate_linear(n=500)
    #x, y = generate_XOR_easy()
    #print(y[2][0])
    model = init_network()
    #print(model)
    for i in range(30000):
        pred, cache = forward(x, model)
        j = cost(pred, y)
        #print(pred)
        acc = compute_acc(pred, y)
        print(j, acc)
        grad = back_prop(model, cache, y)
        r = 0;
        if i<2000:
            r = 0.3
        elif i < 10000:
            r = 0.3
        elif i<20000:
            r = 0.2
        else:
            r = 0.1
        model = update_weight(model, grad, 0.3)
    show_result(x,y,pred)
    print(pred)
    

