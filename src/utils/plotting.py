
# Plotting functions
def clean_ax(ax, clean_all : bool = False, inverted : bool = False):
    """
    Cleans the borders of a matplotlib.axis object.
    Parameters
    ----------
    ax: matplotlib.axis
        axis object to be modified
    clean_all: boolean (default = False)
        whether to clean the entire boundary
    inverted: boolean (default = False)
        whether to clean bottom and right borders
    """
    if clean_all:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)    
        ax.spines['bottom'].set_visible(False)
    elif inverted:
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
    else:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    return ax