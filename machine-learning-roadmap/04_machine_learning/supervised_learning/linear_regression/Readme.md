# Linear Regression — Definitions & Formula Cheat Sheet

## Goal

Linear Regression is a supervised machine learning algorithm used to learn the relationship between an **input (X)** and an **output (Y)** by fitting the best possible straight line through the data.

The model tries to learn:

$$
\hat{Y} = wX + b
$$

where **w** and **b** are learned from the training data.

---

# Terminology

## 1. Feature (X)

The input given to the model.

Examples:

* House size
* Years of experience
* Temperature
* Age

Example:

```python
X = [1000, 1200, 1500, 1800]
```

Meaning:

| House Size |
| ---------- |
| 1000       |
| 1200       |
| 1500       |
| 1800       |

---

## 2. Target / Label (Y)

The correct answer corresponding to every input.

Example:

```python
Y = [200, 240, 300, 360]
```

Meaning:

| House Size | Price |
| ---------- | ----- |
| 1000       | 200   |
| 1200       | 240   |
| 1500       | 300   |
| 1800       | 360   |

The model **never invents Y**. It is provided in the training data.

---

## 3. Prediction (Ŷ)

The value predicted by our model.

Formula:

$$
\hat{Y} = wX + b
$$

Python:

```python
prediction = w * X + b
```

If

```
w = 2
b = 1
```

and

```
X = 5
```

then

```
Prediction = 2×5 + 1 = 11
```

---

## 4. Weight (w)

Weight is the **slope of the line**.

It tells us

> "How much should Y change if X increases by 1?"

Example:

```
w = 2
```

means

```
Increase X by 1

↓

Prediction increases by 2
```

---

## 5. Bias (b)

Bias is the **intercept**.

It shifts the entire line up or down.

Example:

```
Prediction = 2X + 5
```

Even when

```
X = 0
```

prediction becomes

```
5
```

---

## 6. Error (Residual)

Difference between prediction and actual answer.

Formula:

$$
\text{Error} = \hat{Y} - Y
$$

Python:

```python
error = prediction - Y
```

Example

Actual

```
200
```

Prediction

```
180
```

Error

```
-20
```

Positive error → predicted too much.

Negative error → predicted too little.

---

## 7. Loss

Loss measures **how wrong the entire model is**.

We usually use Mean Squared Error (MSE).

Formula:

$$
\text{Loss} = \frac{1}{n} \sum (\text{Error})^2
$$

Python:

```python
loss = np.mean(error ** 2)
```

Example

Errors

```
2
3
-1
```

Square

```
4
9
1
```

Average

```
(4+9+1)/3

=4.67
```

Lower loss means a better model.

---

## 8. Mean Squared Error (MSE)

MSE is the average of squared errors.

Reasons for squaring:

* Prevents positive and negative errors from cancelling each other.
* Penalizes large mistakes more heavily.

Formula:

$$
\text{MSE} = \frac{1}{n} \sum (Y - \hat{Y})^2
$$

---

## 9. Gradient

Gradient tells us

> "Which direction should we move to reduce the loss?"

Without gradients, we know only **how wrong** the model is, not **how to improve it**.

---

## 10. dw

Gradient of the loss with respect to the weight.

Formula:

$$
dw = \frac{\partial \text{Loss}}{\partial w} = \frac{2}{n} \sum (\text{Error} \times X)
$$

Implementation:

```python
dw = (2 / n) * np.sum(error * X)
```

Meaning:

```
How should weight change?
```

If

```
dw > 0
```

decrease the weight.

If

```
dw < 0
```

increase the weight.

---

## 11. db

Gradient of the loss with respect to the bias.

Formula:

$$
db = \frac{\partial \text{Loss}}{\partial b} = \frac{2}{n} \sum \text{Error}
$$

Implementation:

```python
db = (2 / n) * np.sum(error)
```

Meaning:

```
How should bias change?
```

---

## 12. Learning Rate (α)

The size of each step taken during optimization.

Formula:

$$
\alpha
$$

Python:

```python
learning_rate = 0.01
```

Small value:

```
Slow learning
```

Large value:

```
May overshoot the optimum and fail to converge
```

---

## 13. Gradient Descent

Algorithm used to minimize the loss.

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

## 14. Epoch

One complete pass over the entire training dataset.

If

```
100 samples
```

then

```
Epoch 1

↓

Model sees all 100 samples once.
```

---

## 15. Training

The process of repeatedly:

* Predicting
* Computing loss
* Computing gradients
* Updating weights

until the loss becomes small.

---

## 16. Inference

Using the trained model to make predictions on new, unseen data.

Example:

```python
prediction = w * 1600 + b
```

No learning happens during inference.

---

# Complete Training Pipeline

```
Dataset
   │
   ▼
Initialize Weight & Bias
   │
   ▼
Prediction
   │
   ▼
Error
   │
   ▼
Loss
   │
   ▼
Compute dw and db
   │
   ▼
Update Weight & Bias
   │
   ▼
Repeat
   │
   ▼
Final Model
   │
   ▼
Predict New Data
```

---

# Formula Summary

### Model

$$
\hat{Y} = wX + b
$$

### Error

$$
\text{Error} = \hat{Y} - Y
$$

### Loss

$$
\text{Loss} = \frac{1}{n} \sum (\text{Error})^2
$$

### Weight Gradient

$$
dw = \frac{\partial \text{Loss}}{\partial w} = \frac{2}{n} \sum (\text{Error} \times X)
$$

### Bias Gradient

$$
db = \frac{\partial \text{Loss}}{\partial b} = \frac{2}{n} \sum \text{Error}
$$

### Weight Update

$$
w = w - \alpha \cdot dw
$$

### Bias Update

$$
b = b - \alpha \cdot db
$$

---

# One Epoch in Code

```python
predictions = w * X + b

error = predictions - Y

loss = np.mean(error ** 2)

dw = (2 / n) * np.sum(error * X)
db = (2 / n) * np.sum(error)

w = w - learning_rate * dw
b = b - learning_rate * db
```

This loop is the core of linear regression trained with gradient descent.
