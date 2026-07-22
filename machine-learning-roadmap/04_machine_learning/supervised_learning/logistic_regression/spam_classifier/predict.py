# pyrefly: ignore [missing-import]
from data import X, Y
# pyrefly: ignore [missing-import]
from model import LogisticRegression

model = LogisticRegression()

model.train(X, Y)

while True:

    links = float(input("Number of links : "))

    probability = model.predict_probability(links)
    prediction = model.predict(links)

    print(f"Probability : {probability:.4f}")

    if prediction == 1:
        print("Spam\n")
    else:
        print("Not Spam\n")