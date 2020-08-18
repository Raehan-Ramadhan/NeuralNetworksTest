import NeuralNetworks as nn
import numpy as np

with np.load('mnist.npz') as data:
  training_images = data['training_images']
  training_labels = data['training_labels']

training_inputs = [[[0],[1]],[[1],[0]],[[0],[0]],[[1],[1]]]
training_outputs = [[[1],[0]],[[1],[0]],[[0],[1]],[[0],[1]]]

layer_sizes = (784,16,10)

net = nn.NeuralNetworks(layer_sizes)

print(net.weights)
#for a,b in training_inputs,training_outputs
#  net.train(training_images, training_labels, 0.1)