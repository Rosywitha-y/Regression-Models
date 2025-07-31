import pandas as pd
import matplotlib.pyplot as plt
import kagglehub
from kagglehub import KaggleDatasetAdapter

# data processing
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "abrambeyer/openintro-possum",
    "possum.csv"
)

print(df.columns)
df = df.dropna(subset=['taill', 'totlngth'])
points = list(zip(df['taill'], df['totlngth']))

# linear regression
def loss_function(m, b, points):
    total_error = 0
    for x, y in points:
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for x, y in points:
        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))
    
    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b

m = 0
b = 0
L = 0.0001
epochs = 100

for i in range(epochs):
    if i % 50 == 0:
        print(f"Epoch {i}: Loss = {loss_function(m, b, points)}")
    m, b = gradient_descent(m, b, points, L)
print(f"m: {m}, b: {b}")

plt.scatter(df['taill'], df['totlngth'])
plt.xlabel('Tail Length')
plt.ylabel('Total Length')
plt.title('Tail Length vs Total Length of Possums')
plt.plot(df['taill'], m * df['taill'] + b, color='red')
plt.show()