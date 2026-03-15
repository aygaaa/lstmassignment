import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# LSTM Parameters from the PDF
Wf = 0.5; Whf = 0.1; bf = 0
Wi = 0.6; Whi = 0.2; bi = 0
Wc = 0.7; Whc = 0.3; bc = 0
Wo = 0.8; Who = 0.4; bo = 0

Wy = 4; by = 0

# Input sequence
X = [1, 2, 3, 4] # Including 4 as per the PDF's prediction step calculation

# Initial states
h_prev = 0  # h_{t-1}
C_prev = 0  # C_{t-1}

print("--- Manual LSTM Implementation ---")

for t, x_t in enumerate(X):
    print(f"\nTime Step t={t+1}, Input x_t={x_t}")

    # Forget Gate
    f_t = sigmoid(Wf * x_t + Whf * h_prev + bf)
    print(f"Forget Gate (f_t): {f_t:.3f}")

    # Input Gate
    i_t = sigmoid(Wi * x_t + Whi * h_prev + bi)
    print(f"Input Gate (i_t): {i_t:.3f}")

    # Candidate Cell State
    C_tilde_t = tanh(Wc * x_t + Whc * h_prev + bc)
    print(f"Candidate Cell State (C_tilde_t): {C_tilde_t:.3f}")

    # Cell State Update
    C_t = f_t * C_prev + i_t * C_tilde_t
    print(f"Cell State (C_t): {C_t:.3f}")

    # Output Gate
    o_t = sigmoid(Wo * x_t + Who * h_prev + bo)
    print(f"Output Gate (o_t): {o_t:.3f}")

    # Hidden State Update
    h_t = o_t * tanh(C_t)
    print(f"Hidden State (h_t): {h_t:.3f}")

    # Update for next time step
    h_prev = h_t
    C_prev = C_t

# Final Prediction
# The PDF uses the hidden state from the last step (t=4) to predict the value.
y_hat = Wy * h_prev + by
print(f"\n--- Final Prediction ---")
print(f"Predicted next value (y_hat): {y_hat:.3f}")
print(f"Note: The PDF result is ~3.796. Small differences may arise due to rounding in the PDF's manual steps.")
