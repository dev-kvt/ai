import numpy as np

# X = House Size
X = np.array([1, 2, 3, 4], dtype=float)

# Y = House Price
Y = np.array([2, 4, 6, 8], dtype=float)

learning_rate = 0.1
epochs = 10



# ayo weights 
weight = 0 
bias = 0 


n = len(X)


for e in range(epochs):


    predictions = weight * X + bias

    error = predictions - Y 

    loss = np.mean(error**2)

    dweights = (2 / n) * np.sum(error * X)
    dbias = (2 / n) * np.sum(error)


    weight -= learning_rate*dweights
    bias -= learning_rate * dbias
    
    print(f"\nEpoch {e+1}")

    print("Predictions :", predictions)
    print("Errors      :", error)

    print("Loss        :", loss)

    print("dw          :", dweights)
    print("db          :", dbias)

    print("Updated w   :", weight)
    print("Updated b   :", bias)
    print("\n--------------------")
    print("Final Weight :", weight)
    print("Final Bias   :", bias)

    x = 5
    prediction = weight * x + bias

    print("Prediction :", prediction)
