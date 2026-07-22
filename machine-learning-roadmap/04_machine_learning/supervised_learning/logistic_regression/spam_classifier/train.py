# pyrefly: ignore [missing-import]
from data import X, Y
# pyrefly: ignore [missing-import]
from model import LogisticRegression

model = LogisticRegression()

model.train(X, Y)

print("\nTraining Finished\n")

print("Weight :", model.weight)
print("Bias   :", model.bias)

