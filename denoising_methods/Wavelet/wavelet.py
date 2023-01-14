import numpy as np
from PIL import Image
from skimage.restoration import denoise_wavelet


def wavelet_denoising(img):
    img = np.array(Image.open(img), dtype=float) / 255.0

    im_bayes = denoise_wavelet(img, channel_axis=-1, convert2ycbcr=True,
                               method='BayesShrink', mode='soft',
                               rescale_sigma=True)
    im_bayes = np.clip(im_bayes, 0, 1)
    return im_bayes
