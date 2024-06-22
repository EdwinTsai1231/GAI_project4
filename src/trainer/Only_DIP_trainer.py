import torch
import shutil
from .ATrainer import ATrainer
from matplotlib import pyplot as plt

class Only_DIP_Trainer(ATrainer):
  
    def __init__(
        self,
        num_epochs: int,
        original_image,
        noisy_images,
        prev_noisy_images,
        model,
        criterion,
        optimizer,
        device, 
        out_name = "best_model.pth"
    ):
    
        super().__init__()
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.model = model
        self.noisy_images = noisy_images
        self.out_name = out_name
        self.prev_noisy_images = prev_noisy_images
        self.original_image = original_image
        self.device = device
        
    def train(self):
        train_history = {
            "loss": [],
            "noise_psnr": [],
            "ori_psnr": []
        }

        best_record = {
            "best_epoch": 0,
            "noise_psnr": 0,
            "ori_psnr": 0,
        }

        for epoch in range(self.num_epochs):
        
            noisy_image = self.noisy_images[-1]
            prev_noisy_image = self.prev_noisy_images[0]
            
            output = self.model(prev_noisy_image["image_pt"])
            loss = self.criterion(output, noisy_image["image_pt"])
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self._save_output(output, epoch, noisy_image["noise_stage"])

            out2noise_psnr = self._cal_psnr(noisy_image["image_pt"], output)
            out2ori_psnr = self._cal_psnr(self.original_image, output)
            
            train_history["loss"].append(loss.item())
            train_history["noise_psnr"].append(out2noise_psnr)
            train_history["ori_psnr"].append(out2ori_psnr)


            stage_level = noisy_image["noise_stage"]
            
            print(
                f"Epoch [{epoch+1}/{self.num_epochs}], ",
                f"Noise Level: {stage_level}, ",
                f"Loss: {loss.item()}, ",
                f"out2noise_PSNR: {out2noise_psnr}"
            )


            if train_history["ori_psnr"][-1] > best_record["ori_psnr"]:
                best_record["ori_psnr"] = train_history["ori_psnr"][-1]
                best_record["best_epoch"] = epoch
                best_model_state = self.model.state_dict()
        
        e = best_record["best_epoch"]
        shutil.copy2(f'./output_image/output_{e}_{1}.png', './best_picture.png')
        best_epoch = best_record["best_epoch"] + 1
        best_psnr = best_record["ori_psnr"]
        
        print(
            f"Finish training; best stopping point: Epoch {best_epoch} PSNR: {best_psnr}\n",
            f"Automatically save the model in file {self.out_name}."    
        )

        self._plot_psnr_history(train_history['ori_psnr'], train_history['noise_psnr'], best_record)
        torch.save(best_model_state, self.out_name)


    def _plot_psnr_history(self, ori_psnr, noise_psnr, best_record):

        best_psnr = best_record["ori_psnr"]
        best_epoch = best_record["best_epoch"]
        epoch_count = range(1, len(ori_psnr) + 1)
        plt.plot(epoch_count, ori_psnr, 'r-')
        plt.plot(epoch_count, noise_psnr, 'b--')
        plt.legend([f'best: {best_psnr:.3}, epoch: {best_epoch}'])
        plt.xlabel('Epoch')
        plt.ylabel('PSNR')
        plt.savefig('./result.png')
        plt.show()

    