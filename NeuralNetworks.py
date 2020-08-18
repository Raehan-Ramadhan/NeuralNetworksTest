import numpy as np

class NeuralNetworks:
  
  def __init__(self, layer_sizes):
    weight_shapes = [(a,b) for a,b in zip(layer_sizes[1:],layer_sizes[:-1])]
    self.weights = [np.random.standard_normal(s)/s[1]**.5 for s in weight_shapes]
    self.biases = [np.zeros((s,1)) for s in layer_sizes[1:]]

  def predict(self, a):
    for w,b in zip(self.weights,self.biases):
      self.hidden = a
      a = self.sigmoid(np.matmul(w,a) + b)
    return a

  def feedforward(self, input):
    return self.predict(input)

  def train(self, input, target, learning_rate):

    # Generate output
    outputs = self.predict(input)
    
    # Calculate error
    output_error = np.subtract(target, outputs)

    # Calculate gradient
    gradients = np.matrix([self.dsigmoid(a) for a in outputs])
    gradients = np.multiply(gradients, output_error)
    gradients = np.multiply(gradients, learning_rate)

    # Calculate deltas
    hidden_T = np.transpose(self.hidden)
    weight_ho_deltas = np.multiply(gradients, hidden_T)

    # Adjust the weights by deltas
    self.weights[1] = np.add(self.weights[1], weight_ho_deltas)
    # Adjust bias by its deltas(which is the gradient)
    self.biases[1] = np.array(np.add(self.biases[1], gradients))

    # Calculate the hidden layer errors
    who_t = np.transpose(self.weights[1])
    hidden_error = np.multiply(who_t, np.transpose(output_error))

    # Calculate hidden gradient
    hidden_gradients = np.matrix([self.dsigmoid(a) for a in self.hidden])
    hidden_gradients = np.multiply(hidden_gradients, hidden_error)
    hidden_gradients = np.multiply(hidden_gradients, learning_rate)

    # Calculate input->hidden deltas
    inputs_T = np.transpose(input)
    weight_ih_deltas = np.multiply(hidden_gradients, inputs_T)

    # Adjust the weights by deltas
    self.weights[0] = np.add(self.weights[0], weight_ih_deltas)
    # Adjust bias by its deltas(which is the gradient)
    self.biases[0] = np.array(np.add(self.biases[0], hidden_gradients))

    #average_cost = [np.average(a) for a in np.transpose(cost)[0]]
    #hidden_weights = [np.transpose(a) for a in self.weights[1:]]
    #hidden_weights_cost = np.multiply(hidden_weights,average_cost)
    #hidden_cost = np.array([np.sum(a) for a in hidden_weights_cost[0]])
    #output_dsigmoid = map(self.dsigmoid, predictions)
    #multiply()
    
    #numcorrect = sum([np.argmax(a) == np.argmax(b) for a,b in zip(outputs, target)])
    #print('{0}/{1} accuracy: {2}%'.format(numcorrect, len(images), (numcorrect/len(images))*100))

  @staticmethod
  def sigmoid(x):
    return 1/(1+np.exp(-x))

  def dsigmoid(self, y):
    return y * (1 - y)