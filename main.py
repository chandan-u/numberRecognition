
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
        self.weights = [np.random.randn(y,x) for x,y in zip( sizes[:1], sizes[1:])]



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

           training_Data: actual training data
           epochs = number of times to train over a data set in terms of mini batches
           mini_batch_size : The size of the mini bataches that we use for stochaistic gradient descent
           learning_rate: To speed up the gradient descent and travel downhill faster. 

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
               we are calculating the room mean square . The square of the differene is computed for all the  input training data and summed and then averaged.( 1/2n)

            
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


        for epoch in range(epochs):

        	



        

    def costFunction(self, computed_network_output, expected_output):

      	return  computed_output - expected_output 


def sigmoid(z):
    return 1.0/(1.0+ np.exp(-z))



    
    		



net = Network([1,2,3])

print net.biases

print net.weights




