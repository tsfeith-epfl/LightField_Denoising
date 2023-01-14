import numpy as np
import scipy.io as sio
from PIL import Image
from scipy.interpolate import interp1d


def inv_sampling(x, bin_width=0.00001, test_samples=500):
    dist_x, mids = dist(x, bin_width)
    cumulative = np.cumsum(dist_x)
    cumulative = cumulative - np.amin(cumulative)
    f = interp1d(cumulative / np.amax(cumulative), mids)
    return f


def dist(x, bin_width=0.00001):
    hist, bin_edges = np.histogram(x,
                                   bins=np.linspace(np.amin(x), np.amax(x), int((np.amax(x) - np.amin(x)) / bin_width)))
    hist = hist / x.size
    mids = bin_edges[:-1] + np.diff(bin_edges) / 2
    return hist, mids


def load_param():
    matdir_b = './dataset_var_samples_q.mat'
    mat_b = sio.loadmat(matdir_b)
    intrcpt_r = mat_b['b_array_R_d']
    intrcpt_g = mat_b['b_array_G_d']
    intrcpt_b = mat_b['b_array_B_d']

    matdir_a_slope_r = './full_dataset_parameter_R.mat'
    matdir_a_slope_g = './full_dataset_parameter_G.mat'
    matdir_a_slope_b = './full_dataset_parameter_B.mat'
    # R parameters
    mat_a_slope_r = sio.loadmat(matdir_a_slope_r)
    a_r = mat_a_slope_r['a_array_R_d']
    m_r = mat_a_slope_r['slope_array_R']
    a_r = a_r[a_r > 0]
    # G parameters
    mat_a_slope_g = sio.loadmat(matdir_a_slope_g)
    a_g = mat_a_slope_g['a_array_G_d']
    m_g = mat_a_slope_g['slope_array_G']
    a_g = a_g[a_g > 0]
    # B parameters
    mat_a_slope_b = sio.loadmat(matdir_a_slope_b)
    a_b = mat_a_slope_b['a_array_B_d']
    m_b = mat_a_slope_b['slope_array_B']
    a_b = a_b[a_b > 0]

    f_intrcpt_r = inv_sampling(intrcpt_r, test_samples=intrcpt_r.size)
    f_intrcpt_g = inv_sampling(intrcpt_g, test_samples=intrcpt_g.size)
    f_intrcpt_b = inv_sampling(intrcpt_b, test_samples=intrcpt_b.size)

    f_m_r = inv_sampling(m_r, test_samples=m_r.size)
    f_m_g = inv_sampling(m_g, test_samples=m_g.size)
    f_m_b = inv_sampling(m_b, test_samples=m_b.size)

    f_a_r = inv_sampling(a_r, test_samples=a_r.size)
    f_a_g = inv_sampling(a_g, test_samples=a_g.size)
    f_a_b = inv_sampling(a_b, test_samples=a_b.size)

    return f_intrcpt_r, f_m_r, f_a_r, f_intrcpt_g, f_m_g, f_a_g, f_intrcpt_b, f_m_b, f_a_b


def sample_param(f_intercept, f_slope, f_a, n_samples=1):
    intercept = f_intercept(np.random.random(n_samples))
    slope = f_slope(np.random.random(n_samples))
    a = f_a(np.random.random(n_samples))
    b = slope * a + intercept
    return a, b


def sample_param_rgb(f_intrcpt_r, f_m_r, f_a_r, f_intrcpt_g, f_m_g, f_a_g, f_intrcpt_b, f_m_b, f_a_b):
    repeat = True
    while repeat:
        a_r, b_r = sample_param(f_intrcpt_r, f_m_r, f_a_r)
        if b_r > 0:
            repeat = False

    repeat = True
    while repeat:
        a_g, b_g = sample_param(f_intrcpt_g, f_m_g, f_a_g)
        if b_g > 0:
            repeat = False

    repeat = True
    while repeat:
        a_b, b_b = sample_param(f_intrcpt_b, f_m_b, f_a_b)
        if b_b > 0:
            repeat = False

    a = np.array([a_r[0], a_g[0], a_b[0]])
    b = np.array([b_r[0], b_g[0], b_b[0]])

    return a, b


def add_noise(img, a_array, b_array):
    img_dim = img.ndim
    if img_dim == 2:
        ch_n = 1.0
        # print("Warning: Code is for m_g noise")
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        print(img)
    else:
        ch_n = img.shape[2]
    z = np.zeros(img.shape)
    for i in np.arange(0, ch_n, dtype=int):
        y = img[:, :, i]
        a = a_array[i]
        b = b_array[i]
        if a == 0:  # no Poissonian component
            z_i = y
        else:  # % Poissonian component
            chi = 1. / a
            z_i = np.random.poisson(np.maximum(0, chi * y)) / chi

        z_i = z_i + np.sqrt(np.maximum(0, b)) * np.random.normal(loc=0.0,
                                                                 scale=1.0,
                                                                 size=y.shape)  # % Gaussian component
        z[:, :, i] = z_i
    # clipping
    z = np.clip(z, 0.0, 1.0)
    return z


def to_image_from_array(a):
    return Image.fromarray((a * 255.0).round().clip(0, 255).astype(np.uint8))


"""
def add_noise(img, params):
    f_intrcpt_r, f_m_r, f_a_r, f_intrcpt_g, f_m_g, f_a_g, f_intrcpt_b, f_m_b, f_a_b = params
    a, b = sample_param_rgb(f_intrcpt_r, f_m_r, f_a_r, f_intrcpt_g, f_m_g, f_a_g, f_intrcpt_b, f_m_b, f_a_b)
    img_syn_noisy = apply(img, a, b)
    img_syn_noisy_q = (img_syn_noisy * 255.0).round().clip(0, 255).astype(np.uint8)
    return img_syn_noisy
"""
