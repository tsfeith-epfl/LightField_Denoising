import os

import numpy as np


def gridify_images(images):
    """
    Gridify images
    """
    # Get grid size
    grid_size = get_grid_size(images)
    # Gridify images
    grid = []
    for i in range(grid_size[1]):
        row = []
        for j in range(grid_size[0]):
            row.append(images[i * grid_size[0] + j])
        grid.append(row)
    return grid


def get_grid_size(images):
    """
    Get grid size
    """
    # Get grid size from first and last images
    first_image = images[0]
    last_image = images[-1]
    # Get image names
    first_image_name = os.path.basename(first_image)
    last_image_name = os.path.basename(last_image)
    # Get image numbers
    h0, w0 = first_image_name.split('_')[-2], first_image_name.split('_')[-1].split('.')[0]
    h1, w1 = last_image_name.split('_')[-2], last_image_name.split('_')[-1].split('.')[0]
    # Get grid size
    grid_size = [int(w1) - int(w0) + 1, int(h1) - int(h0) + 1]
    return grid_size


def load_numpy_data(grid):
    output = []
    for row in grid:
        output_row = []
        for image in row:
            output_row.append(np.load(image))
        output.append(output_row)
    return output
