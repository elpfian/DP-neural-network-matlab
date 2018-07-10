# Neural Network v1
School project: neural network coded using Matlab, can use with MNIST data

Invoke function: neural_net( inputs, targets, nodeLayers, numEpochs, batchSize, eta )

Output will be written to file results.txt.


### DESCRIPTION

The primary function neural_net:
- Contains the main loop over the epochs, which checks whether the conditions to terminate training have been met (number of epochs or accuracy)
- Generates randomly-sampled mini-batches
- Generates the output of the entire network using the updated weights and biases
- Calculates Mean Square Error, correct classification and accuracy. These are written to a file along with the epoch number for each epoch completed.
- Calls init_weights and init_biases to initialize weights and bias matrices, respectively
- Calls back_prop to complete backpropagation for the network
- Calls sigma and meanSqrErr in computing network output and MSE

The functions init_weights and init_biases:
- Generate initial weight matrix or bias vector for each layer of the network
- Weights and biases are randomly generated with normal distribution, mean of 0 and standard deviation of 1

The function back_prop:
- Performs fully matrix-based backpropagation over each mini-batch
- Returns updated weight and bias matrices for each layer of the network after the mini-batch
- Calls sigma and sigma_prime for feedforward and backpropagation

The functions sigma, sigma_prime and meanSqErr:
- Calculate logsig, logsig derivative and mean squared error, respectively
- Sigma_prime calls sigma
