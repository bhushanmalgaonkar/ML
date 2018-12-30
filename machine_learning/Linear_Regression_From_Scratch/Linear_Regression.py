# Credit: Siraj Raval: Linear Regression Machine Learning (tutorial) (https://www.youtube.com/watch?v=uwwWVAgJBcM)

import numpy as np
import math
from Animate_Linear_Regression import animate


class LinearRegression:
    def __init__(self):
        self.points = []
        self.training_history = {}
        self.learned_parameters = {}

    def error(self, points, b, m):
        # error for line(b, m) is made of difference between actual/expected y and the y given by line (b, m) i.e. b + mx = (y - (b + mx))
        # squares are taken to avoid negative differences and put more penalty on larger errors
        # e = (1/N) * sum[ ( y - (b + mx) )^2 ] for all points

        N = points.shape[0]
        return (1/N) * np.sum(np.square(points[:, 1] - (b + m * points[:, 0])))

    def gradient_step(self, points, b, m):
        N = points.shape[0]
        delta_b = -(2/N) * np.sum(points[:, 1] - (b + m * points[:, 0]))
        delta_m = -(2/N) * np.sum((points[:, 1] - (b + m * points[:, 0])) * points[:, 0])
        return delta_b, delta_m

    def fit(self, points, learning_rate=0.0001, num_iterations=1000, initial_parameters=None):
        # extreme initial condition, line almost perpendicular to expected regression line
        # ideally choose both in the range -1, 1
        if initial_parameters is not None:
            b, m = initial_parameters['b'], initial_parameters['m']
        else:
            b, m = 0, 0

        print('Starting gradient descent from b =\t{0}, m =\t{1}, error =\t{2}'.format(
            b, m, self.error(points, b, m)))

        self.training_history['errors'] = [self.error(points, b, m)]
        self.training_history['bs'] = [b]
        self.training_history['ms'] = [m]

        for itr in range(num_iterations):
            delta_b, delta_m = self.gradient_step(points, b, m)
            b -= learning_rate * delta_b
            m -= learning_rate * delta_m
            err = self.error(points, b, m)
            print('Gradient descent step =\t{0}: b =\t{1}, m =\t{2}, error =\t{3}'.format(
                itr + 1, b, m, err))

            self.training_history['errors'].append(err)
            self.training_history['bs'].append(b)
            self.training_history['ms'].append(m)

        self.learned_parameters = {'b': b, 'm': m}

    def predict(self, x):
        return self.learned_parameters['b'] + self.learned_parameters['m'] * x


if __name__ == '__main__':
    points = np.genfromtxt('data.csv', delimiter=',')

    # tuned parameters
    learning_rate = 0.0004
    num_iterations = 500000
    initial_parameters = {'b': 20, 'm': 0}

    lr = LinearRegression()
    lr.fit(points, learning_rate, num_iterations, initial_parameters)
    animate(lr.training_history, points)

    print('learned_parameters: ', lr.learned_parameters)
    print('Predicted value for x = 10, ', lr.predict(10))
