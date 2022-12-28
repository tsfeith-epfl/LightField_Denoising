from PIL import Image
import numpy as np

from skimage.restoration import denoise_wavelet, estimate_sigma


def wavelet_denoising(img):
    img = np.array(Image.open(img), dtype=float) / 255.0

    im_bayes = denoise_wavelet(img, channel_axis=-1, convert2ycbcr=True,
                               method='BayesShrink', mode='soft',
                               rescale_sigma=True)
    return im_bayes
