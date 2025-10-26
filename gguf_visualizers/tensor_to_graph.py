"""
Tensor to Graph Visualization

Turn a one-dimensional tensor into a bar graph

"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def tensor_to_graph(
        *,
        model_name: str,
        tensor_name: str, 
        tensor: np.ndarray, 
        type_: tuple[str, str],
        central_tendency: float, 
        deviation: float,
        ) -> Image.Image:
    """Convert a tensor to a bar chart visualization as a PIL Image.

    Creates a bar chart where values are color-coded based on their relationship
    to a central tendency and deviation threshold. Values beyond +1 deviation are
    green, values below -1 deviation are red, and values within the range are gray.

    Args:
        tensor_name (str): Name of the tensor to display in the chart title.
        tensor (np.ndarray): 1D numpy array containing the tensor values to visualize.
        central_tendency (float): Reference value (e.g., mean, median) for comparison.
        deviation (float): Threshold value for determining outliers from central tendency.

    Returns:
        Image.Image: PIL Image object containing the generated bar chart visualization.

    Example:
        >>> import numpy as np
        >>> data = np.array([1.0, 2.5, 0.8, 3.2, 1.8])
        >>> image = tensor_to_graph("weights", data, 1.86, 0.8)
        >>> image.save("tensor_chart.png")
    """
    matplotlib.use('Agg')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    indices = np.arange(len(tensor))

    # Pre-calculate colors based on deviation
    colors = []
    for value in tensor:
        if value > central_tendency + deviation:
            colors.append('green')
        elif value < central_tendency - deviation:
            colors.append('red')
        else:
            colors.append('gray')

    # Create bars with pre-calculated colors
    ax.bar(indices, tensor, color=colors, alpha=0.7, width=1.0)

    # Reference lines
    ax.axhline(y=central_tendency, color='blue', linestyle='--', 
               alpha=0.7, label=type_[0].capitalize())
    ax.axhline(y=central_tendency + deviation, color='green', 
               linestyle=':', alpha=0.5, label=f'+1 {type_[1]}')
    ax.axhline(y=central_tendency - deviation, color='red', 
               linestyle=':', alpha=0.5, label=f'-1 {type_[1]}')

    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.set_title(f'Model: {model_name}\nTensor: {tensor_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    buf = buf[:, :, :3]
    plt.close(fig)

    return Image.fromarray(buf, "RGB")