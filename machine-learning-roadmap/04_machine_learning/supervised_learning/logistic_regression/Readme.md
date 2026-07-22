# Logistic Regression — Definitions & Formula Cheat Sheet

## Goal

Logistic Regression is a supervised machine learning algorithm used for **binary classification** (predicting outcomes like 0 or 1, Yes or No, Spam or Not Spam).

Instead of fitting a straight line to directly predict a continuous value, it outputs a **probability between 0 and 1** by passing the linear equation through the **Sigmoid function**.

The model tries to learn:

$$
\hat{Y} = \sigma(z) = \frac{1}{1 + e^{-(wX + b)}}
$$

where **w** (weight) and **b** (bias) are learned from the training data.

---

# Terminology

## 1. Feature (X)

The input given to the model.

Examples:

* Number of links in an email
* Patient metrics (age, blood pressure)
* Customer activity history

Example:

```python
X = [0, 1, 2, 3, 4, 5, 6]
```

Meaning:

| Number of Links |
| --------------- |
| 0               |
| 1               |
| 2               |
| 3               |
| 4               |
| 5               |
| 6               |

---

## 2. Target / Label (Y)

The correct binary class (0 or 1) corresponding to every input.

Example:

```python
Y = [0, 0, 0, 1, 1, 1, 1]
```

Meaning:

| Number of Links | Spam (1) / Not Spam (0) |
| --------------- | ----------------------- |
| 0               | 0                       |
| 1               | 0                       |
| 2               | 0                       |
| 3               | 1                       |
| 4               | 1                       |
| 5               | 1                       |
| 6               | 1                       |

The model **never invents Y**. It is provided in the training data.

---

## 3. Linear Combination (z)

The weighted sum of inputs plus bias (same as the linear regression formula).

Formula:

$$
z = wX + b
$$

Python:

```python
z = w * X + b
```

`z` can range from $-\infty$ to $+\infty$.

---

## 4. Sigmoid Function (σ)

An S-shaped mathematical activation function that maps any real-valued number `z` into a probability score between **0 and 1**.

Formula:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Python:

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

Properties:

* If $z \to +\infty$, $\sigma(z) \to 1$
* If $z = 0$, $\sigma(z) = 0.5$
* If $z \to -\infty$, $\sigma(z) \to 0$

---

## 5. Predicted Probability (Ŷ)

The output of the model representing the probability that the given input $X$ belongs to Class 1.

Formula:

$$
\hat{Y} = \sigma(wX + b) = \frac{1}{1 + e^{-(wX + b)}}
$$

Python:

```python
probability = sigmoid(w * X + b)
```

---

## 6. Decision Boundary & Classification Threshold

To make a final decision (class 0 or class 1), we apply a threshold (usually **0.5**) to the predicted probability.

Formula:

$$
\text{Prediction} = \begin{cases} 1 & \text{if } \hat{Y} \ge 0.5 \\ 0 & \text{if } \hat{Y} < 0.5 \end{cases}
$$

Python:

```python
prediction = 1 if probability >= 0.5 else 0
```

---

## 7. Weight (w)

Weight controls the steepness and direction of the decision curve.

* A **positive weight** means as $X$ increases, probability of class 1 increases.
* A **negative weight** means as $X$ increases, probability of class 1 decreases.

---

## 8. Bias (b)

Bias shifts the Sigmoid curve left or right along the X-axis, setting the baseline probability when $X = 0$.

---

## 9. Error (Residual)

The difference between the predicted probability and the actual target label.

Formula:

$$
\text{Error} = \hat{Y} - Y
$$

Python:

```python
error = prediction - Y
```

---

## 10. Binary Cross-Entropy Loss (Log Loss)

Mean Squared Error is non-convex for logistic regression, so we use **Binary Cross-Entropy Loss (Log Loss)** to measure performance.

Formula for a single sample:

$$
\text{Loss}_i = -\left[ Y_i \log(\hat{Y}_i) + (1 - Y_i) \log(1 - \hat{Y}_i) \right]
$$

Formula for dataset average:

$$
\text{Loss} = -\frac{1}{n} \sum_{i=1}^{n} \left[ Y_i \log(\hat{Y}_i) + (1 - Y_i) \log(1 - \hat{Y}_i) \right]
$$

Python:

```python
loss = -np.mean(Y * np.log(prediction + 1e-9) + (1 - Y) * np.log(1 - prediction + 1e-9))
```

*(Adding a small epsilon `1e-9` prevents `log(0)` errors).*

---

## 11. dw

Gradient of the Log Loss with respect to the weight.

Formula:

$$
dw = \frac{\partial \text{Loss}}{\partial w} = \frac{1}{n} \sum (\hat{Y} - Y) X = \frac{1}{n} \sum (\text{Error} \times X)
$$

Implementation:

```python
dw = (1 / n) * np.sum(error * X)
```

---

## 12. db

Gradient of the Log Loss with respect to the bias.

Formula:

$$
db = \frac{\partial \text{Loss}}{\partial b} = \frac{1}{n} \sum (\hat{Y} - Y) = \frac{1}{n} \sum \text{Error}
$$

Implementation:

```python
db = (1 / n) * np.sum(error)
```

---

## 13. Learning Rate (α)

Controls how large of a step we take in the direction of the negative gradient during optimization.

Python:

```python
learning_rate = 0.1
```

---

## 14. Gradient Descent

Algorithm used to update parameters $w$ and $b$ to minimize Log Loss.

Update rules:

$$
w = w - \alpha \cdot dw
$$

$$
b = b - \alpha \cdot db
$$

Python:

```python
w = w - learning_rate * dw
b = b - learning_rate * db
```

---

## 15. Epoch

One complete iteration over the entire training dataset during which predictions, loss, gradients, and parameter updates are computed.

---

## 16. Training

Iteratively adjusting $w$ and $b$ across multiple epochs until loss is minimized.

---

## 17. Inference

Using the trained model ($w$ and $b$) to classify new, unseen data points.

Python:

```python
probability = sigmoid(w * new_X + b)
prediction = 1 if probability >= 0.5 else 0
```

---

# Complete Training Pipeline

```
Dataset (X, Y)
   │
   ▼
Initialize Weight (w) & Bias (b)
   │
   ▼
Compute z = wX + b
   │
   ▼
Apply Sigmoid: Ŷ = 1 / (1 + e^-z)
   │
   ▼
Compute Log Loss
   │
   ▼
Compute Gradients dw & db
   │
   ▼
Update w & b (Gradient Descent)
   │
   ▼
Repeat for N Epochs
   │
   ▼
Final Model (Trained w & b)
   │
   ▼
Predict Probabilities & Classify New Data
```

---

# Formula Summary

### Model & Prediction

$$
z = wX + b
$$

$$
\hat{Y} = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

### Binary Cross-Entropy Loss (Log Loss)

$$
\text{Loss} = -\frac{1}{n} \sum \left[ Y \log(\hat{Y}) + (1 - Y) \log(1 - \hat{Y}) \right]
$$

### Weight Gradient

$$
dw = \frac{\partial \text{Loss}}{\partial w} = \frac{1}{n} \sum (\text{Error} \times X)
$$

### Bias Gradient

$$
db = \frac{\partial \text{Loss}}{\partial b} = \frac{1}{n} \sum \text{Error}
$$

### Parameter Updates

$$
w = w - \alpha \cdot dw
$$

$$
b = b - \alpha \cdot db
$$

---

# One Epoch in Code

```python
# 1. Forward Pass
z = weight * X + bias
predictions = 1 / (1 + np.exp(-z))

# 2. Compute Loss
loss = -np.mean(Y * np.log(predictions + 1e-9) + (1 - Y) * np.log(1 - predictions + 1e-9))

# 3. Compute Gradients
error = predictions - Y
dw = (1 / n) * np.sum(error * X)
db = (1 / n) * np.sum(error)

# 4. Update Parameters
weight -= learning_rate * dw
bias -= learning_rate * db
```

This loop is the core of logistic regression trained with gradient descent.
