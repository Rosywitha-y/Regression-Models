import math
import numpy as np
import pandas as pd

# data processing
df = pd.read_csv('gender.csv')
feature_cols = ['long_hair',
                'forehead_width_cm',
                'forehead_height_cm',
                'nose_wide',
                'nose_long',
                'lips_thin',
                'distance_nose_to_lip_long']
x = df[feature_cols].to_numpy().astype(float)               # was: x = ...
y = (df['gender'].astype(str).str.lower()
     .map({'male': 0, 'female': 1})
     .to_numpy()
     .reshape(-1, 1)                                        # <— make it (m,1)
     .astype(float))

# weights and bias initialization
n_features = x.shape[1]
w = np.zeros((n_features, 1), dtype=float)
b = 0.0

# z is the dot product of vector of w and x, plus b. z is to be put into da sigmoid
z = np.dot(x, w) + b

def sigmoid_function(z):
    # avoids overflow by clipping
    z = np.clip(z, -500, 500)
    return 1 / ( 1 + np.exp(- z))

# forward pass
def forward(x, w, b):
    z = x @ w + b 
    y_hat = sigmoid_function(z)
    return y_hat # predicted result

# classification loss function
def binary_cross_entropy(y, y_hat, eps=1e-12):
    y_hat = np.clip(y_hat, eps, 1 - eps)  # avoids log(0), idk chatgpt added this??
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

# calculating the gradients for this loss function
def compute_grads(x, y, y_hat):
    m = x.shape[0]
    error = y_hat - y
    dw = (x.T @ error) / m 
    db = np.sum(error) / m
    return dw, db

# training loop
def train_loop(x, y, lr=0.1, epochs=300, w_init=None, b_init=0.0, l2=0.0):
    m, n = x.shape
    w = np.zeros((n, 1)) if w_init is None else w_init.astype(float).copy()
    b = float(b_init)
    history = []
    for t in range(epochs):
        y_hat = forward(x, w, b)    # prediction
        loss = binary_cross_entropy(y, y_hat)   # the loss


        if l2 > 0:
            loss += (l2 / (2 * m)) * np.sum(w * w)  # prevent overfitting
        
        dw, db = compute_grads(x, y, y_hat) # the gradient up or down
        
        if l2 > 0:
            dw += (l2 / m) * w      # prevent overfitting

        # updating w and b
        w -= lr * dw    
        b -= lr * db

        # logging the lost
        if t % 100 == 0 or t == epochs - 1:
            history.append((t, loss))
    return w, b, history

# BRO NOW WE FINALLY START TRAINING UGGHAHAHHAH
w, b, history = train_loop(
    x, y,
    lr=0.1,          
    epochs=2000,
    w_init=w,
    b_init=b,
    l2=0.0
)

# data for human eyes
print("Last 5 loss:", history[-5:])

new_row = np.array([[  # shape (1, 7)
    1,          # long_hair
    12,       # forehead_width_cm
    6,        # forehead_height_cm
    0,          # nose_wide
    0,          # nose_long
    0,          # lips_thin
    1           # distance_nose_to_lip_long
]], dtype=float)

new_prob = forward(new_row, w, b)[0, 0]
new_pred = int(new_prob >= 0.5)
print(f"Input sample: calculated probability = {new_prob:.4f}, prediction = {new_pred}")



