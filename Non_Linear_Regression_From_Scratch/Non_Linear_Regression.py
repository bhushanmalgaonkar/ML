import numpy as np
import math
from Animate_Non_Linear_Regression import animate
from Utils import number_of_lines, get_last_line


class LinearRegression:
    def __init__(self):
        self.x = None
        self.y = None
        self.num_params = None
        self.dataset_size = None
        self.training_history = {}
        self.learned_parameters = None

    def save_progress(self):
        if len(self.training_history['params']) == 0:
            return

        print('Saving progress..')
        base = number_of_lines('training_statistics')
        with open('training_statistics', 'a') as ts:
            for i, (p, e) in enumerate(zip(self.training_history['params'], self.training_history['errors'])):
                ts.write('{0}, {1}, {2}\n'.format(base + i, np.array_str(p), str(e)))
        self.training_history['params'] = []
        self.training_history['errors'] = []

    def load_progress(self, filepath):
        print('Loading progress..')
        last_line = get_last_line(filepath)
        if last_line is None:
            print('Nothing to load')
            return -1, None
        step, params, err = last_line.split(',')
        print('Loading step={0}, params={1}'.format(step, params))
        return int(step), np.fromstring(params.strip('[ '), sep=' ')

    def error(self, params):
        hx = np.zeros(self.dataset_size)
        for i, p in enumerate(params):
            hx += p * (self.x ** i)
        return (1/self.dataset_size) * np.sum(np.square(self.y - hx))

    def gradient_step(self, params):
        hx = np.zeros(self.dataset_size)
        for i, p in enumerate(params):
            hx += p * (self.x ** i)

        delta = np.zeros(self.num_params)
        for i in range(self.num_params):
            delta[i] = -(2/self.dataset_size) * np.sum((self.y - hx) * (self.x ** i))
        return delta

    def fit(self, x, y, degree, learning_rate=0.0001, num_iterations=1000, initial_parameters=None):
        self.x = x
        self.y = y
        self.num_params = int(degree) + 1       # added 1 to handle bias term
        self.dataset_size = x.size

        step, params = self.load_progress('training_statistics')
        if step == -1:
            step = 0
            params = np.zeros(self.num_params)
            if initial_parameters is not None:
                if initial_parameters.size != self.num_params:
                    raise Exception('Number of initial parameters should be `1 + degree`. Size expected {0}, found {1}'.format(
                        self.num_params, initial_parameters.size))
                params = initial_parameters

        self.training_history = {
            'errors': [],
            'params': []
        }

        for itr in range(step, num_iterations):
            delta = self.gradient_step(params)
            params -= learning_rate * delta
            err = self.error(params)

            self.training_history['params'].append(params)
            self.training_history['errors'].append(err)

            # periodic save
            if itr % 10000 == 0:
                print('Gradient descent step =\t{0}: params =\t{1}, error =\t{2}'.format(
                    itr, params, err))
                self.save_progress()

            if math.isinf(err):
                raise Exception('Adjust hyper parameters')

        self.learned_parameters = params
        self.save_progress()

    def predict(self, x):
        hx = 0
        for i, p in enumerate(self.learned_parameters):
            hx += p * (x ** i)
        return hx


if __name__ == '__main__':
    points = np.genfromtxt('data.csv', delimiter=',', skip_header=1)
    x = points[:, 0]
    y = points[:, 1]

    # tuned parameters
    learning_rate = 1e-14
    num_iterations = 500000
    degree = 3
    initial_parameters = np.array([0, 0, 0, 0], dtype=float)

    lr = LinearRegression()
    lr.fit(x, y, degree, learning_rate, num_iterations, initial_parameters)
    #animate(lr.training_history, points)

    print('learned_parameters: ', lr.learned_parameters)
    print('Predicted value for x = 10, ', lr.predict(10))
