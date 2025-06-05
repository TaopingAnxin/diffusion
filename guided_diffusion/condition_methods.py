from abc import ABC, abstractmethod
import torch
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from skimage.metrics import peak_signal_noise_ratio
from scipy.ndimage.filters import convolve
from scipy import ndimage
from scipy import signal
from scipy.signal import convolve2d
import tensorflow as tf
from guided_diffusion.DMSPRestore import filter_image
import time
import scipy.signal as sig

__CONDITIONING_METHOD__ = {}

def MSE(image1, image2):
    """
    Mean Squared Error
    :param image1: image1
    :param image2: image2
    :rtype: float
    :return: MSE value
    """

    # Calculating the Mean Squared Error
    image1.dtype = np.float32
    image1.dtype = np.float32
    mse = np.mean(np.square(image1 - image2))

    return mse


def PSNR(image1, image2, peak=255):
    """
    Peak signal-to-noise ratio
    :param image1: image1
    :param image2: image2
    :param peak: max value of pixel 8-bit image (255)
    :rtype: float
    :return: PSNR value
    """

    # Calculating the Mean Squared Error
    mse = MSE(image1, image2)

    # Calculating the Peak Signal Noise Ratio
    psnr = 10 * np.log10(peak ** 2 / mse)

    return psnr
def convolve_image(image, kernel, mode='valid'):
    """ Implements color image convolution """
    chs = []
    for d in range(image.shape[2]):
        channel = sig.convolve2d(image[:,:,d], kernel, mode=mode)
        chs.append(channel)
    return np.stack(chs, axis=2)
def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)


def scale_image(im, vlow, vhigh, ilow=None, ihigh=None):
    if ilow is None or ihigh is None:
        ilow = im.min()
        ihigh = im.max()
    imo = (im - ilow) / (ihigh - ilow) * (vhigh - vlow) + vlow
    return imo

class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser
    
    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        
        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError
             
        return norm_grad, norm
   
    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass
    
@register_conditioning_method(name='vanilla')
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t

@register_conditioning_method(name='dae')
class PosteriorSampling_meng(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

    # x_prev, x_0_hat, pseudonoise_scale, not used
    def conditioning(self, x_t, measurement, H_funcs, noise_std, alpha_t, alpha_bar, **kwargs):
        x_t = x_t.clone().to('cuda').requires_grad_(True)
        n = x_t.size(0)

        model = kwargs["model"]
        time = kwargs["time"]
        p_mean_variance = kwargs["p_mean_variance"]

        pa = math.sqrt(alpha_bar)
        pb = math.sqrt(1 - alpha_bar)

        beta_t = 1 - alpha_t

        # Loading the model
        DAE = tf.saved_model.load('DAE')

        params = {}
        params['denoiser'] = DAE
        params['sigma_dae'] = 9 

        res = np.squeeze(x_t)
        res = res.detach().cpu().numpy()
        res = np.transpose(res, (1, 2, 0))

        noise = np.random.normal(0.0, params['sigma_dae'], res.shape).astype(np.float32)
        input_image = tf.cast(res + noise, dtype=tf.float32)

        out = params['denoiser'](np.expand_dims(input_image, axis=0))
        rec = out['output'][0, ...]

        x_np = rec.numpy()
        x_np = np.transpose(x_np, (2, 0, 1))
        x_torch = torch.from_numpy(x_np)
        x_torch = x_torch[np.newaxis, :].to('cuda')

        x0_pred = pa * params['sigma_dae']*x_t + pb * pb *x_torch
        x0_pred = x0_pred /(pa *pa*params['sigma_dae'] +pb * pb)

        #x0_pred[0] = torch.from_numpy(res).to(x_t.device).requires_grad_()
        mat = H_funcs.Ht(measurement - H_funcs.H(x0_pred))

        mat_x = (mat.detach() @ x0_pred.reshape(n, -1).t()).sum()

        grad_term = torch.autograd.grad(mat_x, x_t, retain_graph=True)[0]
        grad_term = grad_term.detach()

        coeff = (1 - alpha_t) / np.sqrt(alpha_t) * 50 # deblur=57、inpainting=90、denoise=3\2.2、SR=400、color=400/250

        x0_pred = x0_pred.detach()

        x_t += self.scale * grad_term * coeff

        return x_t

