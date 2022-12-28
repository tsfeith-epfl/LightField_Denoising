import os
import numpy as np
from PIL import Image
import glob
import utils_patch as patch
import utils_image as util
import matplotlib.pyplot as plt

def resize_all():
    """
    Walk through all files (recursively) in the root directory
    and resize all images to have a width of 256 while keeping
    the aspect ratio.
    """
    root = './'
    for root, dirs, files in os.walk(root):
        for file in files:
            if file.endswith('.png'):
                image = Image.open(os.path.join(root, file))
                width, height = image.size
                new_width = 256
                new_height = int(height * new_width / width)
                image = image.resize((new_width, new_height), Image.ANTIALIAS)
                image.save(os.path.join(root, file))


def get_frankenpatches_all():
    """
    Walk through all files (recursively) in the root directory
    and create the frankenpatches for each image.
    Store them as .npy files.
    """
    root = './'
    path_scenes = util.get_scene_paths(root)
    all_scenes = [patch.gridify_images(scene) for scene in path_scenes]
    scenes = [patch.path_grid_to_array(scene) for scene in all_scenes]
    for k, scene in enumerate(scenes):
        print(all_scenes[k][0][0])
        for i in range(len(scene)):
            for j in range((len(scene[i]))):
                frankenpatch = patch.get_frankenpatches(scene, i, j, 5, 6, 1, 20)
                frankenpatch.append(scene[i][j])
                # show one of the examples of all 7 frankenpatches per scene
                if i == 0 and j == 0:
                    plot = plt.figure()
                    for m in range(7):
                        plot.add_subplot(1, 7, m + 1)
                        plt.imshow(frankenpatch[m])
                    plt.show()
                frankenpatch = np.stack(frankenpatch).reshape(frankenpatch[0].shape[0], frankenpatch[0].shape[1], -1)
                # compute file name from all_scenes
                file_name = os.path.basename(all_scenes[k][i][j])
                file_name = file_name.split('.')[0]
                file_name = file_name + '.npy'
                # get dir name from all_scenes
                dir_name = os.path.dirname(all_scenes[k][i][j])
                dir_name = os.path.join(dir_name, 'frankenpatches')
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                np.save(os.path.join(dir_name, file_name), frankenpatch)
