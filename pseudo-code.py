# Pseudo-code for showing the weight updates using gradient decent
w1 = 0
w0 = 0
learning_rate = 0.01
number_of_iterations = 1000
errors = []

for i in range(number_of_iterations):
  
  # calculate predictions
  y_pred = w1 * X + w0
  
  # Calculate the error ( predicted values - actual values )
  error = y - y_pred
  mse = (1 / number_of_samples) * sum(error ** 2)  # MSE calculation
  errors.append(mse)
  
  # Calculate gradients (Formula from 1a)
  gradient_w1 = -(2/number_of_samples) * sum(error * X)
  gradient_w0 = -(2/number_of_samples) * sum(error)
  
  # Updating the weights 
  w1 = w1 - learning_rate * gradient_w1
  w0 = w0 - learning_rate * gradient_w0
  