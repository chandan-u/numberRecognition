
import numpy as np


class Network(object):


    """
        This class helps us to inititalize a entire
        network on neurons by providing an input list eg:[ 2, 4,5]
        in which each number represents the number of neurons in a layer.

        The size of the list is the number of layers. 

        The weights and biases are initialized for all neurons. Except for first layer 
        for which biases are not initialized as it is the input layer. The weights and biases
        are inititalized using gaussian distribution.  ( numpy.random.randn )


        

    """
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        #self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:1], sizes[1:])]
        self.weights = [ np.random.randn(sizes[x+1],sizes[x]) for x in range(0,len(sizes)-1)]


    def feedforward(self, a):
        
        """
           a is the input vector of acitvations. For a given input vector it gives the output of the entire network.
           The logic below iterates over each layer. Ofcourse not the input layer. The input layer has no biases or weights its just input.
           But ofcourse the input is nothing but the Vector of activations "a"
        """
         
        for b,w in zip(self.biases, self.weights):
        	a = sigmoid( np.dot(w,a) + b)

        return a

     
    def SGD(self, training_data, epochs, mini_batch_size, learning_rate):
        """
           input
           -----
           training_data: actual training data
           epochs = number of times to train over a data set in terms of mini batches
           mini_batch_size : The size of the mini bataches that we use for stochaistic gradient descent
           learning_rate: To speed up the gradient descent and travel downhill faster. 
           -----


           In Stochaistic gradient descent we dont use the entire data to train the network . we randomly pick mini batches of data
           and train the network. The epoch means the number of times we pick the mini batches to train the network.


           In neural networks when you say the network is learning it means it is able to adjust its weights and biases and fine tune
           itself. Thereby reduce the cost/loss/error. In this case the network adjusts its weights and biases using The stochaistic
           Gradient Descent method. 
           
           For the given program we are using the below cost function to evaluate the performance of the network:

               C(w,b) =  1/2n *  Sigma over x ( || y(x)  - a ||)^2

               c(w,b) is the funciton which measures error for given weights w and biases b in the network.
               y(x) is computed output of the network
               a is the expected output.
               we are calculating the root mean square . The square of the differene is computed for all the  input training data and summed and then averaged.( 1/2n)

            
            Now how does this function help the network learn?

              well since the cost function is a measure of error. The least we get the c(w,b) the better is the configuration of the netowrk.
              But how do we change/tune the weights and biases so that the  c(w,b) decreases. Well thats where gradient descent comes in. It uses differentiation to 
              to minimize the cost function. Since the cost function is dependent on variables weights w and biases b. It differentiates over these two variables and obtains the change that must be done to travel towards minimization.

              A simple explanation is : Imagine that the cost function is a ball and it needs to travel downhill towords minimization. 
              In this case the variables or dimensions that are influenicng the cost function are weights w and biases b. We slightly move the ball by changing the weights and biases in small amounts. 
              using differentiation.

            Well why do we need to change the weights in small amoounts. Why cant we skip to the most minma?
              Visually we can see the local minima  in a diagram . But how does a computer know , it doesnt have eyes to point out the minma.
              Hence it uses many iterations of differntiation to reach the minma.

            When do we know that we reached the minma?
              
                Well when the cost function or measure of error of the network approximates to zero or is very minmal.       

           
            Stochaistic gradient Descent vs Gradient Descent?
              Stochaistic means random. 


        """  


        training_size = len(training_data)
        for epoch in range(epochs):

            np.random.shuffle(training_data)
            mini_batches = [ training_data[index: index + mini_batch_size]  for index in xrange(0, training_size, mini_batch_size)]
           
            for mini_batch in mini_batches:
                 self.updateMiniBatch(mini_batch, learning_rate)



    def updateMiniBatch(self, mini_batch, eta):
    	"""
            eta is the learning rate of the gradient descent 
            (to travel faster down hill)
     	"""

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            #print delta_nabla_b, delta_nabla_w
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

        


            
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass

        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())


        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    

    def cost_derivative(self, computed_network_output, expected_output):
      	return  (computed_network_output - expected_output) 



def sigmoid(z):
    return 1.0/(1.0+ np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
    
    		









