import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util
import utils.utils_patch as patch
from PIL import Image


class DatasetDnCNN(data.Dataset):
    """
    # -----------------------------------------
    # Get L/H for denosing on AWGN with fixed sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., DnCNN
    # -----------------------------------------
    """

    def __init__(self, opt):
        super(DatasetDnCNN, self).__init__()
        self.patches_L = None
        self.patches_H = None
        self.patches_names = None
        print('Dataset: Denosing on AWGN with fixed sigma. Only dataroot_H is needed.')
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = opt['H_size'] if opt['H_size'] else 256
        self.frankenpatch_size = opt['frankenpatch_size'] if opt['frankenpatch_size'] else 96
        self.frankenpatch_stride = opt['frankenpatch_stride'] if opt['frankenpatch_stride'] else 2
        self.num_similars = opt['num_similars'] if opt['num_similars'] else 9

        self.path_scenes0 = util.get_scene_paths(opt['dataroot_H'])
        self.path_scenes0 = [patch.gridify_images(scene) for scene in self.path_scenes]
        self.path_scenes0 = np.array([patch.load_numpy_data(scene) for scene in self.path_scenes])
        self.path_scenes = []
        for scene in self.path_scenes0:
            for row in scene:
                for image in row:
                    self.path_scenes.append(image)
        print(len(self.path_scenes))

    def __getitem__(self, index):

        # ------------------------------------
        # get scene
        # ------------------------------------
        imgs_H = self.path_scenes[index]

        if self.opt['phase'] == 'train':
            """
            # --------------------------------
            # get L/H patch pairs
            # --------------------------------
            """
            H, W, _ = imgs_H.shape

            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))

            patch_H = imgs_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # --------------------------------
            # augmentation - flip, rotate
            # --------------------------------
            mode = random.randint(0, 7)
            patch_H = util.augment_img(patch_H, mode)

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_H = util.uint2tensor3(patch_H)
            img_L = img_H.clone()

            # --------------------------------
            # add noise
            # --------------------------------
            self.sigma = random.randint(0, 55)
            noise = torch.randn(img_L.size()).mul_(self.sigma/255.0)
            img_L.add_(noise)

        else:
            """
            # --------------------------------
            # get L/H image pairs
            # --------------------------------
            """
            img_L = np.copy(imgs_H)

            # --------------------------------
            # add noise
            # --------------------------------
            noise_strength = random.randint(0, 2)
            if noise_strength == 0:
                self.sigma = 15
            elif noise_strength == 1:
                self.sigma = 25
            else:
                self.sigma = 50
            img_L += np.random.normal(0, self.sigma/255.0, img_L.shape)

            # -------------------------------
            # HWC to CHW, numpy to tensor
            # --------------------------------
            img_L = util.uint2tensor3(img_L)
            img_H = util.uint2tensor3(imgs_H)

        return {'L': img_L, 'H': img_H[-3:], "SIGMA": self.sigma}
    def __len__(self):
        return len(self.path_scenes)
