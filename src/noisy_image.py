import torch
import numpy as np
from PIL import Image
from copy import deepcopy

class Noisy_Image_Generator:
    
    @staticmethod
    def pil_to_np(img_pil):
        '''
        From W x H x C [0...255] to C x W x H [0..1]
        '''
        ar = np.array(img_pil).transpose(2,0,1)
        
        return ar.astype(np.float32) / 255.

    @staticmethod
    def np_to_pil(img_np): 
        '''
        From C x W x H [0..1] to  W x H x C [0...255]
        '''
        ar = np.clip(img_np*255,0,255).astype(np.uint8)
        ar = ar.transpose(1, 2, 0)

        return Image.fromarray(ar)

    @staticmethod
    def np_to_torch(img_np):
        '''
        From C x W x H [0..1] to  1 x C x W x H [0..1]
        '''
        return torch.from_numpy(img_np)[None, :]

    @staticmethod
    def torch_to_np(img_pt):
        '''
        From 1 x C x W x H [0..1] to C x W x H [0..1]
        '''
        return img_pt.detach().cpu().numpy()[0]
    
    @staticmethod
    def init_noise(input_depth, spatial_size):
        if isinstance(spatial_size, int):
            spatial_size = (spatial_size, spatial_size)
        
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape).normal_() * 1./10
        
        return net_input
    
    def __init__(
        self,
        input_image_path,
        noise_scale,
        num,
        device
    ):
        self.input_image_pil = Image.open(input_image_path)
        self.input_image_np = self.pil_to_np(self.input_image_pil)
        self.noise_scale = noise_scale
        self.num = num
        self.device = device
        
    def generate(self, output_dir = None):
        
        input_image_pt = self.np_to_torch(self.input_image_np)\
                            .type(torch.float32).detach().to(self.device)
        
        image_np = self._add_noise(self.input_image_np, self.noise_scale, 1, output_dir = output_dir)
        image_pt = self.np_to_torch(image_np).to(self.device)
        noisy_images = [
            {
                "image_pt": image_pt,
                "noise_stage": 1
            }
        ]

        for i in range(1, self.num):
            image_np = self.torch_to_np(noisy_images[i-1]["image_pt"])
            image_np = self._add_noise(image_np, self.noise_scale, i, output_dir = output_dir)
            image_pt = self.np_to_torch(image_np).to(self.device)

            noisy_images.append(
                {
                    "image_pt": image_pt,
                    "noise_stage": i+1
                }

            )

        noisy_images.reverse()
        prev_noisy_images = deepcopy(noisy_images[:-1]) 
        prev_noisy_images.insert(
            0,
            {
                "image_pt": self.gen_net_input(),
                "noise_stage": 0
            }
        )
        

        return input_image_pt, prev_noisy_images, noisy_images
    
    def gen_net_input(self):
        net_input = self.init_noise(
            input_depth = 3, 
            spatial_size = (
                self.input_image_pil.size[0], 
                self.input_image_pil.size[1]
            )
        ).type(torch.float32).detach().to(self.device)

        return net_input
    
    def _add_noise(self, img_np, noise_scale, noise_stage, output_dir = None):
        
        img_noisy_np = np.clip(
            img_np + np.random.normal(scale=noise_scale, size=img_np.shape),
            0, 1
        ).astype(np.float32)
            
        if output_dir is not None:
            img_noisy_pil = self.np_to_pil(img_noisy_np)
            img_noisy_pil.save(f'{output_dir}/noise_{noise_stage}.png', format='PNG', compress_level=0)
        

        return img_noisy_np
