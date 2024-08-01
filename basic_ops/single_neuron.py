import math
import numpy as np

def sigmoid(logit: float):
  exp = math.exp(logit)
  sig = exp / (1 + exp)
  return sig

def dot(w: list[float], f: list[float]) -> float:
  res = 0
  for weight, feature in zip(w,f):
    res += weight * feature
    
  return res
	
def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
    probabilities = [0 for x in features]
    error_sum = 0 
    w = weights
    for i, f in enumerate(features):
        interm = dot(w,f) + bias 
        probabilities[i] = sigmoid(interm)
        error_sum += (probabilities[i] - labels[i])**2

    mse = error_sum/len(features)

    return probabilities, mse


def partial_deriv(probability, feature: float, label):
   return 2*(probability-label) * probability * (1-probability) * feature

def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
    # Your code here
    num_examples, num_features = features.shape
    weights = initial_weights
    bias = initial_bias

    mses = [0 for _ in range(epochs)]

    for epoch in range(epochs):
        # probabilities has length num_examples
        probabilities, mse = single_neuron_model(features, labels, weights, bias)
        mses[epoch] = mse
        grad = [0 for i in range(num_features)]
        partial_deriv_bias = 0
        # accumulate gradient for this epoch
        for i in range(num_examples):
            current_example = features[i]
            for j in range(num_features):
                grad[j] += partial_deriv(probabilities[i], current_example[j], labels[i])
            partial_deriv_bias += partial_deriv(probabilities[i], 1, labels[i])
        # perform the gradient update
        for a in range(num_features):
            weights[a] -= learning_rate * grad[a]/num_examples
        
        bias -= learning_rate * partial_deriv_bias

    return weights, bias, mses