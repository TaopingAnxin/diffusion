from functools import partial
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import argparse
import yaml
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clip, clear_color,clear, mask_generator, center_crop
from util.logger import get_logger
from util.data_preprocessing import CenterCropLongEdge

def load_yaml(file_path: str) -> dict:
    with open(file_path,encoding='utf-8') as f:  #此处有修改，with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

import numpy as np
from skimage.metrics import peak_signal_noise_ratio
import random
import os
import time
import subprocess


# for debug
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
seed_torch()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', default='configs/model_config.yaml', type=str)
    parser.add_argument('--diffusion_config', default='configs/diffusion_config.yaml', type=str)
    parser.add_argument('--task_config', default='configs/deblur_uniform_config.yaml', type=str)#此处修改
    
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
 
    seed_torch(args.seed)
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)

    
    # Load model
    model = create_model(**model_config)
   
    if model_config['use_fp16']:
        model.convert_to_fp16()
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
   
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
   
    # Working directory
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'])
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'], 'seed_{}'.format(args.seed))
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'truth']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    if data_config['name'] == 'imagenet' or data_config['name'] == 'cat':
        transform = transforms.Compose([
            CenterCropLongEdge(),
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)


    ## get degradation matrix ##
    H_funcs = None
    if measure_config['operator']['name'] == 'super_resolution':
        if measure_config['operator']['type'] == 'standard':
            ratio = measure_config['operator']['scale_factor']
            from util.functions import SuperResolution
            H_funcs = SuperResolution(3, 256, ratio, device)
        elif measure_config['operator']['type'] == 'bicubic':
            factor = measure_config['operator']['scale_factor']
            ratio = factor
            print('ratio: {}'.format(ratio))
            from util.functions import SRConv
            def bicubic_kernel(x, a=-0.5):
                if abs(x) <= 1:
                    return (a + 2)*abs(x)**3 - (a + 3)*abs(x)**2 + 1
                elif 1 < abs(x) and abs(x) < 2:
                    return a*abs(x)**3 - 5*a*abs(x)**2 + 8*a*abs(x) - 4*a
                else:
                    return 0
            k = np.zeros((factor * 4))
            for i in range(factor * 4):
                x = (1/factor)*(i- np.floor(factor*4/2) + 0.5)
                k[i] = bicubic_kernel(x)
            k = k / np.sum(k)
            kernel = torch.from_numpy(k).float().to(device)
            H_funcs = SRConv(kernel / kernel.sum(),model.in_channels, model.image_size, device, stride = factor)
        else:
            print("ERROR: super_resolution type not supported")
            quit()

    elif measure_config['operator']['name']  == 'color':
        from util.functions import Colorization
        H_funcs = Colorization(model.image_size, device)
        ratio = 1
    elif measure_config['operator']['name'] == 'denoise':
        from util.functions import Denoising
        H_funcs = Denoising(model.in_channels, model.image_size, device)
        ratio = 1
    elif measure_config["operator"]["name"] == "inpainting":
        from util.functions import Inpainting
        if measure_config["operator"]["type"] == "inp_lolcat":
            loaded = np.load("inp_masks/lolcat_extra.npy")
            mask = torch.from_numpy(loaded).to(device).reshape(-1)
            missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3     
        elif measure_config["operator"]["type"] == "inp_freeform_30-40%":
            mask_shape = [10000, 256, 256]
            loaded = dict(np.load("inp_masks/imagenet_freeform_masks.npz"))
            loaded["30-40% freeform"] = (
                np.unpackbits(loaded["30-40% freeform"], axis=None)[
                    : np.prod(mask_shape)
                ]
                .reshape(mask_shape)
                .astype(np.uint8)
            )
            mask = torch.from_numpy(loaded["30-40% freeform"][0]).to(device).reshape(-1)
            missing_r = torch.nonzero(mask == 1).long().reshape(-1) * 3
        else:
            missing_r = (
                torch.randperm(model.image_size**2)[: model.image_size**2 // 2]
                .to(device)
                .long()
                * 3
            )
            # print(missing_r.shape)  # [32768]
        missing_g = missing_r + 1
        missing_b = missing_g + 1
        missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
        H_funcs = Inpainting(model.in_channels, model.image_size, missing, device)
        ratio = 1
    else:
        print("ERROR: The task type not supported")
        quit()

    # Do Inference
    start_time  = time.time()
    Num_count = 0
    psnr_results = []
    for i, ref_img in enumerate(loader):
        logger.info(f"Inference for image {i}")
        fname = str(i).zfill(5) + '.png'
        ref_img = ref_img.to(device)
        Num_count += 1
        
        y_x = H_funcs.H(ref_img) # forward linear measurements

        y_n = y_x + noiser.sigma * torch.randn_like(y_x)

        # print(ref_img.shape)
        # print(y_x.shape)
        # print(y_n.shape)
        
        # General-Purpose Posterior Sampling via DMPS
        DMPS_start_time = time.time()
        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        sample = sample_fn(x_start=x_start, measurement=y_n, H_funcs=H_funcs, noise_std = noiser.sigma, record=True, save_root=out_path)
        DMPS_end_time = time.time()
        print('DMPS running time: {}'.format(DMPS_end_time - DMPS_start_time))
        psnr = peak_signal_noise_ratio(ref_img.cpu().numpy(),sample.cpu().numpy())
        psnr_results.append([psnr])
        print('PSNR: {}'.format(psnr))

        if measure_config['operator']['name']  == 'color':
            y_n = y_n.reshape(1,1,model.image_size,model.image_size)
            plt.imsave(os.path.join(out_path, 'input', fname), clear(y_n),cmap='gray')
        elif measure_config["operator"]["name"] == "inpainting":
            input_size = int(model.image_size / ratio)
            pinv_y_n = H_funcs.H_pinv(y_n).view(
                y_n.shape[0], model.in_channels, input_size, input_size
            )
            pinv_y_n += (
                H_funcs.H_pinv(H_funcs.H(torch.ones_like(pinv_y_n))).reshape(
                    *pinv_y_n.shape
                )
                - 1
            )
            plt.imsave(os.path.join(out_path, "input", fname), clear_color(pinv_y_n))
        else:
            input_size = int(model.image_size/ratio)  
            y_n = y_n.reshape(1,model.in_channels,input_size,input_size)
            plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
        plt.imsave(os.path.join(out_path, 'truth', fname), clear_color(ref_img))
        plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample))

    end_time = time.time()
    running_time = end_time - start_time
    save_results = np.zeros(3)
    save_results[0] = measure_config['noise']['sigma']
    save_results[1] = running_time
    save_results[2] = Num_count
    np.savetxt(os.path.join(out_path, 'saved_results.csv'),save_results)


    np.savetxt(os.path.join(out_path, 'psnr_results.csv'),np.array(psnr_results))
    print('Total # imges:{}, total  running Time: {}'.format(Num_count,end_time - start_time))
    command = ['python', 'util/compute_metric.py']
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT)
        print(output.decode("gb18030"))
    except subprocess.CalledProcessError as e:
        print("命令执行出错:", e.output.decode("gb18030"))
if __name__ == '__main__':
    main()
