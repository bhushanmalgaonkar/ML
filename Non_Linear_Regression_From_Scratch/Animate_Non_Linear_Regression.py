import numpy as np
import math
import matplotlib.pyplot as plt


def plot_curve(params, x_min, x_max):
    x = np.arange(x_min, x_max, 0.1)
    y = np.zeros(x.shape[0])
    for i, p in enumerate(params):
        y += p * (x ** i)
    plt.plot(x, y, c='r')


def animate(training_history_filepath, data_filepath):
    keyframes = []
    key = 1
    with open(training_history_filepath) as f:
        for i, l in enumerate(f):
            if key == i:
                key = math.ceil(i * 1.01)
                params = np.fromstring(l.strip().split(',')[1].strip('[] '), sep=' ')
                keyframes.append(params)
    print(len(keyframes))

    points = np.genfromtxt(data_filepath, delimiter=',', skip_header=1)

    # for drawing fitted line
    MIN_X = -100
    MAX_X = 100

    # for drawing axes around points
    PADDING = 5

    plt.figure(figsize=(15, 12))

    for i, keyframe in enumerate(keyframes):
        print(i, keyframe)
        plt.clf()
        plt.scatter(points[:, 0], points[:, 1], s=10)
        if i > 0 and np.array_equal(keyframes[i - 1], keyframe):
            continue
        plot_curve(keyframe, MIN_X, MAX_X)

        # set axes boundaries as matplotlib changes them to fit entire figure
        plt.xlim(np.min(points[:, 0]) - PADDING, np.max(points[:, 0]) + PADDING)
        plt.ylim(np.min(points[:, 1]) - PADDING, np.max(points[:, 1]) + PADDING)
        plt.pause(0.00001)

    plt.show()


if __name__ == '__main__':
    animate('training_statistics', 'data.csv')
