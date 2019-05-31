import numpy as np
import matplotlib.pyplot as plt

z = np.arange(-3, 3, .1)
zero = np.zeros(len(z))
y = np.max([zero, z], axis=0)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(z, y)
ax.set_ylim([-3.0, 3.0])
ax.set_xlim([-3.0, 3.0])
ax.set_title('ReLU')

plt.show()
