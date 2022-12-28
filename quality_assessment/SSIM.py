from pytorch_msssim import ssim


def SSIM(orig, rec):
    # orig and rec are tensors
    # with values in range [0, 1]
    # and shape [3, H, W]
    if len(orig.shape) != 3 or len(rec.shape) != 3:
        raise ValueError('SSIM expects 3D tensors with shape [3, H, W]')
    if orig.shape[0] != 3 or rec.shape[0] != 3:
        raise ValueError('SSIM expects 3-channel images, don\'t use the Y channel images')

    orig = orig.unsqueeze(0)
    rec = rec.unsqueeze(0)

    return ssim(orig, rec, data_range=1, size_average=True).item()
