# task10-aiml
import pandas as pd
from sklearn.datasets import load_digits

digits = load_digits()

df = pd.DataFrame(digits.data)
df["target"] = digits.target

df.to_csv("digits_dataset.csv", index=False)
print("Dataset created successfully")
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv("digits_dataset.csv")

X = data.drop("target", axis=1)
y = data["target"]

print("Dataset Shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

k_values = [3, 5, 7, 9]
accuracies = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print("Accuracy for K =", k, ":", acc)

plt.plot(k_values, accuracies, marker="o")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("Accuracy vs K")
plt.show()

best_model = KNeighborsClassifier(n_neighbors=3)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
import matplotlib.pyplot as plt
import numpy as np

for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X_test[i].reshape(8, 8), cmap="gray")
    plt.title("Pred: " + str(y_pred[i]))
    plt.axis("off")

plt.show()


