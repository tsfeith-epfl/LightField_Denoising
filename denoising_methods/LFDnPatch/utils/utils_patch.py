from PIL import Image
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import math


def get_frankenpatches(grid, i, j, patch_size, num_similars, search_stride, roi):
    """
    Build stitched images of similar patches from the other SAIs

    Args:
        grid: grid of images
        i: row index of the image
        j: column index of the image
        patch_size: size of the patch
        num_similars: number of similar patches to find
        search_stride: stride of the search
        roi: number of steps to take in each direction
    """
    output = [np.zeros(grid[i][j].shape) for _ in range(num_similars)]
    h_pos = range(0, grid[i][j].shape[0], patch_size)
    w_pos = range(0, grid[i][j].shape[1], patch_size)
    for h in h_pos:
        for w in w_pos:
            matching_patches = get_matching_patches(grid, [i, j], [h, w], patch_size, num_similars, search_stride, roi)
            while len(matching_patches) < num_similars:
                matching_patches += matching_patches
            matching_patches = matching_patches[-num_similars:]
            for k in range(num_similars):
                output[k][h:h+patch_size, w:w+patch_size, :] = matching_patches[k][1]
    return output


def get_matching_patches(grid, grid_pos, start_pos, patch_size, num_patches, search_stride, roi):
    """
    Get matching patches
    """
    patch_size = [min(grid[grid_pos[0]][grid_pos[1]].shape[0] - start_pos[0], patch_size),
                  min(grid[grid_pos[0]][grid_pos[1]].shape[1] - start_pos[1], patch_size)]
    target_patch = grid[grid_pos[0]][grid_pos[1]][start_pos[0]:start_pos[0]+patch_size[0],
                                                  start_pos[1]:start_pos[1]+patch_size[1],
                                                  :]

    grid_h = [[grid_pos[0], i] for i in range(len(grid[0]))]
    grid_v = [[i, grid_pos[1]] for i in range(len(grid))]
    grid_h.remove(grid_pos)
    grid_v.remove(grid_pos)

    matching_patches = []
    prev_position = start_pos
    for position in grid_h:
        min_difference = float('inf')
        new_image = grid[position[0]][position[1]]
        search_space = []
        for step in range(-roi, roi + 1):
            range_h = [prev_position[1] + step * search_stride, prev_position[1] + step * search_stride + patch_size[1]]
            if range_h[0] >= 0 and range_h[1] <= new_image.shape[1]:
                search_space.append(range_h)
        for range_h in search_space:
            candidate_patch = new_image[start_pos[0]:start_pos[0] + patch_size[0], range_h[0]:range_h[1], :]
            difference = np.sum(np.abs(target_patch - candidate_patch))
            if difference < min_difference:
                min_difference = difference
                best_patch = candidate_patch
                prev_position = [start_pos[0], range_h[0]]
        matching_patches = limited_insert(matching_patches, best_patch, min_difference, num_patches)

    prev_position = start_pos
    for position in grid_v:
        min_difference = float('inf')
        new_image = grid[position[0]][position[1]]
        search_space = []
        for step in range(-roi, roi + 1):
            range_v = [prev_position[0] + step * search_stride, prev_position[0] + step * search_stride + patch_size[0]]
            if range_v[0] >= 0 and range_v[1] <= new_image.shape[0]:
                search_space.append(range_v)
        for range_v in search_space:
            candidate_patch = new_image[range_v[0]:range_v[1], start_pos[1]:start_pos[1] + patch_size[1], :]
            difference = np.sum(np.abs(target_patch - candidate_patch))
            if difference < min_difference:
                min_difference = difference
                best_patch = candidate_patch
                prev_position = [range_v[0], start_pos[1]]
        matching_patches = limited_insert(matching_patches, best_patch, min_difference, num_patches)

    return matching_patches


def limited_insert(arr, item, difference, num_patches):
    """
    Insert item into arr with binary search
    """
    if len(arr) < num_patches:
        arr.append([difference, item])
        if len(arr) == num_patches:
            arr.sort(key=lambda x: x[0])
    else:
        if difference < arr[-1][0]:
            left = 0
            right = num_patches - 1
            while left < right:
                mid = (left + right) // 2
                if arr[mid][0] < difference:
                    left = mid + 1
                else:
                    right = mid
            arr.insert(left, [difference, item])
            arr.pop()
    return arr


def get_images(scene_name):
    """
    Get images from a scene
    """
    # Get all images from the scene
    images = glob.glob(os.path.join('../../../data/noisy/phone', scene_name, '*.png'))
    images.sort()
    return images


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


def path_grid_to_array(grid):
    """
    Convert grid of paths to grid of images
    """
    output = []
    for row in grid:
        output_row = []
        for image in row:
            output_row.append(np.array(Image.open(image), dtype=np.float32) / 255.0)
        output.append(output_row)
    return output


def load_numpy_data(grid):
    output = []
    for row in grid:
        output_row = []
        for image in row:
            output_row.append(np.load(image))
        output.append(output_row)
    return output

def add_noise_grid(grid, noise_level, i, j):
    """
    Add noise to a grid
    """
    output = grid.copy()
    for k in range(len(grid)):
        output[k][j] += np.random.normal(0, noise_level, grid[k][j].shape)
    for k in range(len(grid[0])):
        if k != j:
            output[i][k] += np.random.normal(0, noise_level, grid[i][k].shape)
    return output


if __name__ == '__main__':
    from time import perf_counter
    scene_name = 't_rex'
    images = get_images(scene_name)
    grid = gridify_images(images)
    grid = path_grid_to_array(grid)
    start = perf_counter()
    output = get_frankenpatches(grid, 2, 2, 5, 6, 2, 5)
    end = perf_counter()
    print(end - start)
    for index, image in enumerate(output):
        image = Image.fromarray((image * 255).astype('uint8'))
        image.save('./t_rex/t_rex_{}.png'.format(index))
