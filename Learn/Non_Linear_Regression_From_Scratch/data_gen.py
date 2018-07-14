import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

params = [3.5, 0, -0.5, 5.7]
xmin = -50
xmax = 50
dataset_size = 200
noise = 100000

x = xmin + np.random.random(dataset_size) * (xmax - xmin)
y = np.zeros(dataset_size)
for i, p in enumerate(params):
    y += p * x ** i
# noise
y += np.random.random(dataset_size) * noise

data = {
    'x': x,
    'y': y
}
df = pd.DataFrame(data=data)
df.to_csv('data.csv', index=False)

plt.scatter(x, y, s=5)
plt.show()
