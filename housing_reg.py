import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target

def loss_function(m, b, df):
    y_pred = m * df['MedInc'] + b
    return ((df['MedHouseVal'] - y_pred) ** 2).mean()  # Vectorized

def gradient_descent(m_now, b_now, df, L):
    n = len(df)
    y_pred = m_now * df['MedInc'] + b_now
    error = df['MedHouseVal'] - y_pred

    m_gradient = -(2/n) * (df['MedInc'] * error).sum()  # Vectorized
    b_gradient = -(2/n) * error.sum()

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b

m = 0
b = 0
L = 0.0001
epochs = 500

for i in range(epochs):
    if i % 50 ==0:
        print(f"Epoch {i}: Loss = {loss_function(m, b, df)}")
    m, b = gradient_descent(m, b, df, L)

print(f"m: {m}, b: {b}")
plt.scatter(df['MedInc'], df['MedHouseVal'])
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.title('Median Income vs Median House Value in California')
plt.plot(df['MedInc'], m * df['MedInc'] + b, color='red')
plt.show()