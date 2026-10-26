
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


np.random.seed(42)
X = 2 * np.random.rand(100, 1)  
y = 4 + 3 * X + np.random.randn(100, 1)   

# 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)


y_pred = lin_reg.predict(X_test)


print(f' (Slope / Coefficient): {lin_reg.coef_[0][0]:.2f}')
print(f' (Intercept): {lin_reg.intercept_[0]:.2f}')
print(f' y = {lin_reg.coef_[0][0]:.2f}*X + {lin_reg.intercept_[0]:.2f}\n')


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'(MSE): {mse:.2f}')
print(f'  (R2 Score): {r2:.2f} (the nearest to 1 the best is the model)\n')


plt.scatter(X_test, y_test, color='blue', label='real data (Test Data)')
plt.plot(
    X_test, y_pred, color='red', linewidth=2, label= '    (Model)'
)
plt.xlabel('feature (X)')
plt.ylabel('result (y)')
plt.title('Linear Regression Example')
plt.legend()
plt.show()