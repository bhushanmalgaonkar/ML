import numpy as np
import math
import matplotlib.pyplot as plt


def animate(training_history, points=None):
    errs = training_history['errors']
    bs = training_history['bs']
    ms = training_history['ms']

    # for drawing fitted line
    MIN_X = -100
    MAX_X = 200

    # for drawing axes around points
    PADDING = 20

    plt.figure(figsize=(15, 12))

    plt.text(80, 130, "Step: 0")
    plt.scatter(points[:, 0], points[:, 1], s=10)
    plt.plot([MIN_X, MAX_X], [bs[0] + ms[0] * MIN_X, bs[0] + ms[0] * MAX_X], c='r')
    plt.xlim(np.min(points[:, 0]) - PADDING, np.max(points[:, 0]) + PADDING)
    plt.ylim(np.min(points[:, 1]) - PADDING, np.max(points[:, 1]) + PADDING)

    if points is not None:
        for p in points:
            plt.plot([p[0], p[0]], [p[1], bs[0] + ms[0] * p[0]], c='g', lw=0.3)
            plt.xlim(np.min(points[:, 0]) - PADDING, np.max(points[:, 0]) + PADDING)
            plt.ylim(np.min(points[:, 1]) - PADDING, np.max(points[:, 1]) + PADDING)
            plt.pause(0.0001)

    # show animation of only 50 frames
    i = 1
    while i < len(errs):
        print('Drawing {0}/{1}'.format(i, len(errs)))
        plt.clf()

        plt.text(80, 130, "Step: {0}".format(i))
        if points is not None:
            for p in points:
                plt.plot([p[0], p[0]], [p[1], bs[i] + ms[i] * p[0]], c='g', lw=0.3)

        plt.scatter(points[:, 0], points[:, 1], s=10)
        plt.plot([MIN_X, MAX_X], [bs[i] + ms[i] * MIN_X, bs[i] + ms[i] * MAX_X], c='r')

        # as the line regresses towards convergence, change becomes smaller and smaller -> take larger steps for animation
        i = math.ceil(i * 1.1)

        # set axes boundaries as matplotlib changes them to fit entire figure
        plt.xlim(np.min(points[:, 0]) - PADDING, np.max(points[:, 0]) + PADDING)
        plt.ylim(np.min(points[:, 1]) - PADDING, np.max(points[:, 1]) + PADDING)
        plt.pause(0.00001)

    plt.show()
