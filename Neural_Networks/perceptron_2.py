## Example of Perceptron

# Imports
import numpy as np

# Class
class Perceptron:
     #This class models an artificial neuron with step activation function.
    def __init__(self, weights = np.array([1]), threshold = 0):
        """
        Initialize weights and threshold based on input arguments. 
        Note that no type-checking is being performed here for simplicity.
        """
        self.weights = weights.astype(float) 
        self.threshold = threshold

    def activate(self, values):
        """
        Takes in @param values, a list of numbers equal to length of weights.
        @return the output of a threshold perceptron with given inputs based on
        perceptron weights and threshold.
        """

        # First calculate the strength with which the perceptron fires
        strength = np.dot(values, self.weights)
        # Then return 0 or 1 depending on strength compared to threshold  
        return int(strength > self.threshold)


    def update(self, values, train, eta=.1):
        """
        Takes in a 2D array @param values consisting of a LIST of inputs and a
        1D array @param train, consisting of a corresponding list of expected
        outputs. Updates internal weights according to the perceptron training
        rule using these values and an optional learning rate, @param eta.
        """

        # For each data point:
        for data_point in range(len(values)):
            # Obtain the neuron's prediction for the data_point
            prediction = self.activate(values[data_point])

            # Get the prediction accuracy calculated as 
            # (expected value - predicted value)
            error = train[data_point] - prediction

            # Update self.weights based on the multiplication of:
            # - prediction accuracy(error)
            # - learning rate(eta)
            # - input value(values[data_point])
            weight_update = eta * error * values[data_point]
            self.weights += weight_update

def test():
    """
    A few tests to make sure that the perceptron class performs as expected.
    Nothing should show up in the output if all the assertions pass.
    """
    def sum_almost_equal(array1, array2, tol = 1e-6):
        return sum(abs(array1 - array2)) < tol

    p1 = Perceptron(np.array([1,1,1]),0)
    p1.update(np.array([[2,0,-3]]), np.array([1]))
    print(sum_almost_equal(p1.weights, np.array([1.2, 1, 0.7])))

    p2 = Perceptron(np.array([1,2,3]),0)
    p2.update(np.array([[3,2,1],[4,0,-1]]),np.array([0,0]))
    print(sum_almost_equal(p2.weights, np.array([0.7, 1.8, 2.9])))

    p3 = Perceptron(np.array([3,0,2]),0)
    p3.update(np.array([[2,-2,4],[-1,-3,2],[0,2,1]]),np.array([0,1,0]))
    print(sum_almost_equal(p3.weights, np.array([2.7, -0.3, 1.7])))

test()