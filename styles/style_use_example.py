import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
import sys, os

# print current directory:
print('current dir is:' + os.getcwd())

print('MPL configuration directory is: ' + mpl.get_configdir())



plt.style.use("college_paper")  # Load the custom style

# Now create your plot via pandas
# Create a sample DataFrame
data = {
    'x': np.arange(10),  # X values from 0 to 9
    'y': np.random.rand(10) * 10,  # Random Y values
    'z': np.random.rand(10) * 5  # Random Z values
}
df = pd.DataFrame(data)

df.plot()

# Then tweak the y-axis in code
ax = plt.gca()                # get current axes
ax.spines["left"].set_color("red")
ax.tick_params(axis="y", colors="red")

# Show
plt.show()


# Or read from direcory
# plt.style.use("./styles/college_ruled.mplstyle")