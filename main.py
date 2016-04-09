from network import Network

from data import load_data_wrapper




neural_network = Network([784, 30, 10])


training_data, validation_data, test_data = load_data_wrapper()

neural_network.SGD(training_data, 30, 10, 3.0)



