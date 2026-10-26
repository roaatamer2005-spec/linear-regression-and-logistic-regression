import numpy as np
import pandas as pd

np.random.seed(42)


n = 200
class0 = np.random.randn(n, 2) + [-2, -2]
class1 = np.random.randn(n, 2) + [2, 2]
df = pd.DataFrame(np.vstack([class0, class1]), columns=["x1", "x2"])
df["label"] = [0] * n + [1] * n
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle


split = int(0.8 * len(df))
train, test = df[:split], df[split:]
X_train, y_train = train[["x1", "x2"]].values, train["label"].values
X_test, y_test = test[["x1", "x2"]].values, test["label"].values



def sigmoid(z):
    return 1 / (1 + np.exp(-z))


w = np.zeros(X_train.shape[1])
b = 0.0
lr = 0.1

for epoch in range(200):
    p = sigmoid(X_train @ w + b)                       
    loss = -np.mean(y_train * np.log(p + 1e-9) +
                    (1 - y_train) * np.log(1 - p + 1e-9))  
    dw = X_train.T @ (p - y_train) / len(y_train)      
    db = np.mean(p - y_train)
    w -= lr * dw                                       
    b -= lr * db
    if epoch % 40 == 0:
        print(f"epoch {epoch:3d} | loss = {loss:.4f}")


pred = (sigmoid(X_test @ w + b) >= 0.5).astype(int)
accuracy = np.mean(pred == y_test)
tp = np.sum((pred == 1) & (y_test == 1))
fp = np.sum((pred == 1) & (y_test == 0))
fn = np.sum((pred == 0) & (y_test == 1))
print(f"\naccuracy  = {accuracy:.3f}")
print(f"precision = {tp / (tp + fp + 1e-9):.3f}")
print(f"recall    = {tp / (tp + fn + 1e-9):.3f}")
print("weights:", w, "bias:", b)
