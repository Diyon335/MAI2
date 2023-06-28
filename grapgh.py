import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Create a figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define the function to update the graph for each frame
def update(frame):
    ax.clear()  # Clear the previous frame

    # Generate some data points
    t = np.linspace(0, 2*np.pi, 100)
    x = np.cos(t + frame/10)
    y = np.sin(t + frame/10)
    z = t

    # Plot the data
    ax.plot(x, y, z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Moving 3D Graph')

# Create the animation
animation = FuncAnimation(fig, update, frames=100, interval=50)

# Display the animation
plt.show()
