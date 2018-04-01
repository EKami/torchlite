import matplotlib.pyplot as plt
from matplotlib import patches, patheffects
from torchlite.torch.tools import tensor_tools


def draw_img(image, figsize=None, title=None, show=False):
    """
    Draw an image from a numpy array or a torch tensor
    Args:
        image (np.array, torch.Tensor): An image as a numpy array or a tensor
        figsize (tuple, None): The figure size or None for the size to automatically be inferred from the image
        title (str): The image title
        show (bool): If True the image will be shown on screen

    Returns:
        matplotlib.axis.Axis: A matplotlib axis
    """
    image = tensor_tools.to_np(image)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.title.set_text(title)
    if show:
        plt.show()
    return ax


def draw_black_outline(ax, line_width=1):
    """
    Sets black outlines around an axis object.
    Typically useful to display white text, the black outlines will contrast
    the text so it's easier to read it.
    Args:
        ax (matplotlib.axis.Axis): A matplotlib axis
        line_width (int): The width of the line to draw (usually 1 for text)
    """
    ax.set_path_effects([patheffects.Stroke(linewidth=line_width, foreground='black'), patheffects.Normal()])


def draw_rect(ax, xy, width, height):
    """
    Draw a rectangle on a given axis with outlines
    Args:
        ax (matplotlib.axis.Axis): A matplotlib axis
        xy (tuple): The x and y position of the text to draw
        width (int): Rectangle width
        height (int): Rectangle height
    """
    patch = ax.add_patch(patches.Rectangle(xy, width, height, fill=False, edgecolor='white', lw=2))
    draw_black_outline(patch, 4)


def draw_text(ax, xy, text, text_size=14):
    """
    Draw a text on a given axis with outlines
    Args:
        ax (matplotlib.axis.Axis): A matplotlib axis
        xy (tuple): The x and y position of the text to draw
        text (str): The text to draw
        text_size (int): The size of the text
    """
    text = ax.text(*xy, text, verticalalignment='top', color='white', fontsize=text_size, weight='bold')
    draw_black_outline(text, 1)
