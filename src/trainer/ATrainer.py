from abc import ABC, abstractmethod
from src.noisy_image import Noisy_Image_Generator
from skimage.metrics import peak_signal_noise_ratio as psnr

class ATrainer(ABC):
    def __init__(self):
      pass
        
    @abstractmethod
    def train(self):
        raise NotImplementedError


    def _cal_psnr(self, ground_pt, output_pt):
        ground_np = ground_pt.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
        output_np = output_pt.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
        
        psnr_value = psnr(ground_np, output_np)
        
        return psnr_value
    

    def _save_output(self, output, epoch, noise_level):
        saved_np = output.detach().cpu().numpy().squeeze()
        output_pil = Noisy_Image_Generator.np_to_pil(saved_np)
        output_pil.save(
            f'output_image/output_{epoch}_{noise_level}.png',
            format='PNG', compress_level=0
        )