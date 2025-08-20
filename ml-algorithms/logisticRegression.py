#to predict a categorical outcome (like "yes" or "no") 
#we model the probability of that outcome using an S-shaped curve (sigmoid function)
import numpy as np
import matplotlib.pyplot as plt

# A simple dataset for demonstration
X = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.5])
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1])

# Reshape X for matrix operations
X = X.reshape(-1, 1)

# Hyperparameters
learning_rate = 0.01
iterations = 10000

# Initialize the model's parameters (weights)
weights = np.zeros(1)
bias = 0
n = len(X)

# The Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# The core of the algorithm: the Gradient Descent loop
for i in range(iterations):
    linear_output = np.dot(X, weights) + bias
    y_predicted = sigmoid(linear_output)

    dw = (1/n) * np.dot(X.T, (y_predicted - y))
    db = (1/n) * np.sum(y_predicted - y)

    weights = weights - learning_rate * dw
    bias = bias - learning_rate * db
    
print("\nTraining complete!")
print(f"Final weights: {weights[0]:.4f}, Final bias: {bias:.4f}")

# --- VISUALIZATION---

# Create a range of x values to plot the sigmoid curve
x_plot = np.linspace(0, 6, 100).reshape(-1, 1)

# Calculate the predicted probabilities for the plot range
linear_output_plot = np.dot(x_plot, weights) + bias
y_predicted_plot = sigmoid(linear_output_plot)

# 1. Create a scatter plot of the actual data points
plt.scatter(X, y, color='blue', label='Actual Data Points (0=Fail, 1=Pass)')

# 2. Plot the sigmoid curve
plt.plot(x_plot, y_predicted_plot, color='red', label='Logistic Regression Curve')

# 3. Add a horizontal line at 0.5 to show the decision boundary
plt.axhline(y=0.5, color='green', linestyle='--', label='Decision Boundary')

# Add titles and labels for clarity
plt.title('Logistic Regression with Gradient Descent')
plt.xlabel('Hours Studied')
plt.ylabel('Probability of Passing')
plt.legend()
plt.grid(True)
plt.show()


# Make a prediction with the trained model
new_x = np.array([[2.8]])
final_linear_output = np.dot(new_x, weights) + bias
final_prediction_prob = sigmoid(final_linear_output)
final_prediction_class = 1 if final_prediction_prob[0][0] >= 0.5 else 0

print(f"\nProbability of passing for a student who studied 2.8 hours: {final_prediction_prob[0][0]:.4f}")
print(f"Predicted class: {'Pass' if final_prediction_class == 1 else 'Fail'}")