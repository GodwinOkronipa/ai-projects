#Understanding Linear regression using gradient descent
#Used in predicting values that deviate from an expected linear behavior model
import numpy as np  #to handle large arrays and matrices of data much more efficiently than lists
import matplotlib.pyplot as plt  # Import the visualization library 


# A simple dataset for demonstration
# X represents the independent variable (e.g., hours studied)
# y represents the dependent variable (e.g., test scores)
X = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 8, 11])

# Hyperparameters: these are values you can tune to improve the model
learning_rate = 0.01  # Controls the size of the "steps" down the gradient
iterations = 1000     # The number of times the parameters will be updated


# Initialize the models parameters (slope and intercept)
# We start with random or zero values and let the algorithm find the best ones
m = 0
b = 0
n = len(X) # Number of data points

# The core of the algorithm: the Gradient Descent loop
for i in range(iterations):
    # Step 1: Calculate the predictions for the current m and b
    # This is our line equation: y = mx + b
    y_predicted = m * X + b

    # Step 2: Calculate the gradients
    # Gradients are the partial derivatives of the Mean Squared Error (MSE) cost function
    # They tell us the direction and magnitude of the steepest ascent of the error
    
    # Derivative with respect to m (slope)
    # The -2/n part is from the derivative of the MSE formula
    D_m = (-2/n) * np.sum(X * (y - y_predicted))
    
    # Derivative with respect to b (intercept)
    D_b = (-2/n) * np.sum(y - y_predicted)

    # Step 3: Update the parameters
    # We move in the opposite direction of the gradient (downhill)
    # The learning rate controls how big each step is
    m = m - learning_rate * D_m
    b = b - learning_rate * D_b

    #Print the progress every 100 iterations to see the values change
    if i % 100 == 0:
        print(f"Iteration {i}: m = {m:.4f}, b = {b:.4f}, loss = {np.mean((y - y_predicted)**2):.4f}")

# After the loop, the algorithm has found the optimal values for m and b
print("\nTraining complete!")
print(f"Final parameters: m = {m:.4f}, b = {b:.4f}")

# ---VISUALIZATION ---

# 1. Create a scatter plot of the actual data points
plt.scatter(X, y, color='blue', label='Actual Data Points')

# 2. Plot the line of best fit
# We use the final, optimized m and b to draw our line
y_final = m * X + b
plt.plot(X, y_final, color='red', label='Line of Best Fit')

plt.title('Linear Regression with Gradient Descent')
plt.xlabel('X (Independent Variable)')
plt.ylabel('y (Dependent Variable)')
plt.legend()
plt.grid(True)
plt.show() # Display the plot

# Now you can use the trained model to make a prediction: eg
new_x = 6
prediction = m * new_x + b
print(f"For an input of {new_x}, the model predicts a value of {prediction:.4f}"