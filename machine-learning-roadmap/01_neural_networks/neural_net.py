import numpy as np


# ============================================================
# DATA
# ============================================================

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)


# ============================================================
# PARAMETERS
# ============================================================

np.random.seed(42)

# 2 inputs -> 4 hidden neurons
W1 = np.random.randn(2, 4) * 0.5
b1 = np.zeros((1, 4))

# 4 hidden neurons -> 1 output
W2 = np.random.randn(4, 1) * 0.5
b2 = np.zeros((1, 1))


# ============================================================
# ACTIVATION FUNCTIONS
# ============================================================

def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ============================================================
# FORWARD PROPAGATION
# ============================================================

def forward(X):

    # Layer 1
    z1 = X @ W1 + b1
    a1 = relu(z1)

    # Layer 2
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)

    return z1, a1, z2, a2


# ============================================================
# LOSS
# ============================================================

def loss(y, prediction):

    epsilon = 1e-8

    return -np.mean(
        y * np.log(prediction + epsilon)
        + (1 - y) * np.log(1 - prediction + epsilon)
    )


# ============================================================
# BACKPROPAGATION
# ============================================================

def backward(X, y, z1, a1, prediction):

    m = len(X)

    # --------------------------------
    # Output layer
    # --------------------------------

    dz2 = prediction - y

    dW2 = (a1.T @ dz2) / m

    db2 = np.sum(dz2, axis=0, keepdims=True) / m


    # --------------------------------
    # Hidden layer
    # --------------------------------

    da1 = dz2 @ W2.T

    dz1 = da1 * relu_derivative(z1)

    dW1 = (X.T @ dz1) / m

    db1 = np.sum(dz1, axis=0, keepdims=True) / m


    return dW1, db1, dW2, db2


# ============================================================
# TRAINING
# ============================================================

learning_rate = 0.1
epochs = 10000

for epoch in range(epochs):

    # ---------- FORWARD ----------
    z1, a1, z2, prediction = forward(X)

    # ---------- LOSS ----------
    current_loss = loss(y, prediction)

    # ---------- BACKWARD ----------
    dW1, db1, dW2, db2 = backward(
        X,
        y,
        z1,
        a1,
        prediction
    )

    # ---------- UPDATE ----------
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    if epoch % 1000 == 0:
        print(
            f"Epoch {epoch} | Loss: {current_loss:.4f}"
        )


# ============================================================
# TEST
# ============================================================

_, _, _, predictions = forward(X)

print("\nPredictions:")

for input_value, prediction in zip(X, predictions):

    print(
        input_value,
        "=>",
        prediction[0],
        "=>",
        int(prediction[0] >= 0.5)
    )
