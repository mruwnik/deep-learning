import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output


def graph_updater(y_min=0, y_max=1):
    """Get a function to display and update a graph.

    The updater function keeps a list of all the values
    that were passed to it, and each new item will be added
    as the rightmost point on the graph.
    
    The y axis is limited between <y_min, y_max>, by default
    displaying values between <0, 1>.
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1) 

    def add_datapoint(y):
        xs, ys = [], []
        if ax.lines:
            xs = ax.lines[0].get_xdata()
            ys = ax.lines[0].get_ydata()

        prev_x = xs[-1] if len(xs) > 0 else 0        

        ax.cla()
        ax.plot(np.append(xs, prev_x + 1), np.append(ys, y))
        
        ax.set_ylim(y_min, y_max)
        
        display(fig)

        clear_output(wait = True)
    
    return add_datapoint
