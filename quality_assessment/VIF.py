from IQA_pytorch import VIFs


def VIF(orig, rec):
    # orig and rec are tensors
    # with values in the range [0, 1]
    # and shape [1, H, W]
    if len(orig.shape) != 3 or len(rec.shape) != 3:
        raise ValueError('VIF expects 3D tensors with shape [1, H, W]')
    if orig.shape[0] != 1 or rec.shape[0] != 1:
        raise ValueError('VIF expects 1-channel images, use `utilities.rgb_to_one_channel` to convert them to 1-channel')
    vif = VIFs(channels=1)
    orig = orig.unsqueeze(0)
    rec = rec.unsqueeze(0)
    return vif(orig, rec, as_loss=False).mean().item()
