import time
import numpy as np
import scipy.signal as sig
from PIL import Image


def compute_PSNR_pad(img1, img2, pad_y, pad_x):
    """ Computes peak signal-to-noise ratio between two images. 
    Input:
    img1: First image in range of [0, 255].
    img2: Second image in range of [0, 255].
    pad_y: Scalar radius to exclude boundaries from contributing to PSNR computation in vertical direction.
    pad_x: Scalar radius to exclude boundaries from contributing to PSNR computation in horizontal direction.
    
    Output: PSNR """

    img1_u = (np.clip(np.squeeze(img1), 0, 255.0)[pad_y:-pad_y,pad_x:-pad_x,:]).astype(dtype=np.uint8)
    img2_u = (np.clip(np.squeeze(img2), 0, 255.0)[pad_y:-pad_y,pad_x:-pad_x,:]).astype(dtype=np.uint8)
    imdiff = (img1_u).astype(dtype=np.float32) - (img2_u).astype(dtype=np.float32)
    rmse = np.sqrt(np.mean(np.power(imdiff[:], 2)))
    return 20.0 * np.log10(255.0 / rmse)

def compute_PSNR(img1, img2, kernel_shape):
    pad_y = np.floor(kernel_shape[0] / 2.0).astype(np.int64)
    pad_x = np.floor(kernel_shape[1] / 2.0).astype(np.int64)
    
    return compute_PSNR_pad(img1, img2, pad_y, pad_x)

def pad_image(img, kernel_shape):
    pad_y = np.floor(kernel_shape[0] / 2.0).astype(np.int64)
    pad_x = np.floor(kernel_shape[1] / 2.0).astype(np.int64)
    
    return np.pad(img, pad_width=((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode='edge').astype(np.float32)

def filter_image(image, kernel, mode='valid'):
    """ Implements color filtering (convolution using a flipped kernel) """
    chs = []
    for d in range(image.shape[2]):
        channel = sig.convolve2d(image[:,:,d], np.flipud(np.fliplr(kernel)), mode=mode)
        chs.append(channel)
    return np.stack(chs, axis=2)

def convolve_image(image, kernel, mode='valid'):
    """ Implements color image convolution """
    chs = []
    for d in range(image.shape[2]):
        channel = sig.convolve2d(image[:,:,d], kernel, mode=mode)
        chs.append(channel)
    return np.stack(chs, axis=2)


def DMSP_restore(degraded, kernel, subsampling_mask, sigma_d, params):
    """ Implements stochastic gradient descent (SGD) Bayes risk minimization for image restoration described in:
     "Deep Mean-Shift Priors for Image Restoration" (http://home.inf.unibe.ch/~bigdeli/DMSPrior.html)
     S. A. Bigdeli, M. Jin, P. Favaro, M. Zwicker, Advances in Neural Information Processing Systems (NIPS), 2017 
     
     Input:
     degraded: Observed degraded RGB input image in range of [0, 255].
     kernel: Blur kernel width odd dimentions(internally flipped for convolution).
     subsampling_mask: The (sub/down sampling) mask indicating the used values in the degraded input 
     sigma_d: Noise standard deviation. (set to 0 for noise-blind deblurring)
     params: Set of parameters.
     params.denoiser: The denoiser function hanlde.
    
     Optional parameters:
     params.sigma_dae: The standard deviation of the denoiser training noise. default: 11
     params.num_iter: Specifies number of iterations.
     params.mu: The momentum for SGD optimization. default: 0.9
     params.alpha the step length in SGD optimization. default: 0.1
    
     Outputs:
     res: Solution."""
    
    def run_dae(input_image):
        out = params['denoiser'](np.expand_dims(input_image, axis=0))
        return out['output'][0,...]
    
    if 'denoiser' not in params:
        raise ValueError('Need a denoiser in params.denoiser!')
        
    if 'gt' in params:
        print_iter = True
    else:
        print_iter = False
    
    if 'sigma_dae' not in params:
        params['sigma_dae'] = 11.0
        
    if 'num_iter' not in params:
        params['num_iter'] = 10
        
    if 'mu' not in params:
        params['mu'] = 0.9
    
    if 'alpha' not in params:
        params['alpha'] = 0.1
    
    
    #     intitiate the result image by filling-in the blanks

    tmp = convolve_image(degraded, np.ones((3,3)), mode='same')
    weights = convolve_image(subsampling_mask, np.ones((3,3)), mode='same')
    tmp = degraded*subsampling_mask + (tmp/np.maximum(weights, 1e-5))*(1.0-subsampling_mask)
    res = pad_image(tmp, kernel.shape)

    #     init the optimization step with zeros
    step = np.zeros(res.shape)

    if print_iter:
        psnr = compute_PSNR(params['gt'], res, kernel.shape)
        print ('Initialized with PSNR: ' + str(psnr))

    for iter in range(params['num_iter']):
        if print_iter:
            print('Running iteration: ' + str(iter))
            t = time.time()

        #     compute prior gradient
        noise = np.random.normal(0.0, params['sigma_dae'], res.shape).astype(np.float32)
        rec = run_dae(res + noise)
        prior_grad = res - rec

        #     compute data gradient
        map_conv = filter_image(res, kernel)
        data_err = subsampling_mask *(map_conv - degraded) * (subsampling_mask.size/subsampling_mask.sum())
        data_grad = convolve_image(data_err, kernel, mode='full')

        relative_weight = 0.5
        if sigma_d <= 0:
            sigma2 = 2 * params['sigma_dae'] * params['sigma_dae']
            lambda_ = (degraded.size) / (
                        np.sum(np.power(data_err[:], 2)) + subsampling_mask.sum() * sigma2 * (np.sum(np.power(kernel[:], 2))))
            relative_weight = lambda_ / (lambda_ + 1 / params['sigma_dae'] / params['sigma_dae'])
        else:
            relative_weight = (1 / sigma_d / sigma_d) / (
                        1 / sigma_d / sigma_d + 1 / params['sigma_dae'] / params['sigma_dae'])

        #     sum the gradients
        grad_joint = data_grad * relative_weight + prior_grad * (1 - relative_weight);

        #     update the results
        step = params['mu'] * step - params['alpha'] * grad_joint;
        res = res + step;
        res = np.minimum(255.0, np.maximum(0, res)).astype(np.float32);

        if print_iter:
            psnr = compute_PSNR(params['gt'], res, kernel.shape)
            print ('PSNR is: ' + str(psnr) + ', iteration finished in ' + str(time.time() - t) + ' seconds')

    return res