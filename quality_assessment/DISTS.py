from IQA_pytorch import DISTS as D


def DISTS(orig, rec):
    # orig and rec are tensors
    # with values in the range [0, 1]
    # and shape [3, H, W]
    if len(orig.shape) != 3 or len(rec.shape) != 3:
        raise ValueError('VIF expects 3D tensors with shape [3, H, W]')
    if orig.shape[0] != 3 or rec.shape[0] != 3:
        raise ValueError('DISTS expects RGB images.')
    dists = D(channels=3)
    orig = orig.unsqueeze(0)
    rec = rec.unsqueeze(0)
    return dists(orig, rec, as_loss=False).mean().item()
