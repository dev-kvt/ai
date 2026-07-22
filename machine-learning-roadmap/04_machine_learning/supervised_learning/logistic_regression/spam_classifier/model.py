import numpy as np 

class LogisticRegression:
    def __init__(self):
        self.weight = 0.0
        self.bias = 0.0

    def sigmoid(self , z ):
        return 1 / (1 + np.exp(-z))

    
    def train(self , X ,Y , learning_rate = 0.1 , epochs = 5000):

        n = len(X)


        for e in range(epochs):

            # forward pass 

            z = self.weight * X + self.bias
            prediction = self.sigmoid(z)


            # loss func 
            loss = -np.mean(
                Y * np.log(prediction + 1e-9)
                + (1 - Y) * np.log(1 - prediction + 1e-9)
            )
                        # Gradients
            dw = (1 / n) * np.sum((prediction - Y) * X)
            db = (1 / n) * np.sum(prediction - Y)

            # Update
            self.weight -= learning_rate * dw
            self.bias -= learning_rate * db

            if e % 500 == 0:
                print(f"Epoch {e} Loss = {loss:.4f}")

    def predict_probability(self, x):

        z = self.weight * x + self.bias
        return self.sigmoid(z)

    def predict(self, x):

        probability = self.predict_probability(x)

        if probability >= 0.5:
            return 1
        return 0

