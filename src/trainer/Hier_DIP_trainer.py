import torch
import shutil
from skimage.metrics import peak_signal_noise_ratio as psnr
from matplotlib import pyplot as plt

class Hier_DIP_Trainer:
  
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
    train_history_30 = {
      "loss": [],
      "noise_psnr": [],
      "ori_psnr": []
    }

    train_history_60 = {
      "loss": [],
      "noise_psnr": [],
      "ori_psnr": []
    }

    train_history_90 = {
      "loss": [],
      "noise_psnr": [],
      "ori_psnr": []
    }

    train_history_99 = {
      "loss": [],
      "noise_psnr": [],
      "ori_psnr": []
    }

    best_epoch = 0
    best_psnr = 0

    for epoch in range(self.num_epochs):
      
      total_loss = 0

      for i, (noisy_image, prev_noisy_image) in enumerate(zip(self.noisy_images, self.prev_noisy_images)):
        
        output = self.model(prev_noisy_image["image_pt"])

        criterion_loss = self.criterion(output, noisy_image["image_pt"])
        total_loss = criterion_loss


        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()


        if i == 99:
          self._save_output(output, epoch, noisy_image["noise_stage"])


        total_loss += total_loss.item()

        out2noise_psnr = self._cal_psnr(noisy_image["image_pt"], output)
        out2ori_psnr = self._cal_psnr(self.original_image, output)
        
        
       
        if i == 30:
          train_history_30["loss"].append(total_loss.item())
          train_history_30["noise_psnr"].append(out2noise_psnr)
          train_history_30["ori_psnr"].append(out2ori_psnr)

        elif i == 60:
          train_history_60["loss"].append(total_loss.item())
          train_history_60["noise_psnr"].append(out2noise_psnr)
          train_history_60["ori_psnr"].append(out2ori_psnr)

        elif i == 90:
          train_history_90["loss"].append(total_loss.item())
          train_history_90["noise_psnr"].append(out2noise_psnr)
          train_history_90["ori_psnr"].append(out2ori_psnr)

        elif i == 99:
          train_history_99["loss"].append(total_loss.item())
          train_history_99["noise_psnr"].append(out2noise_psnr)
          train_history_99["ori_psnr"].append(out2ori_psnr)


        stage_level = noisy_image["noise_stage"]
        print(f"Epoch [{epoch+1}/{self.num_epochs}], Noise Level: {stage_level}, Loss: {total_loss.item()}, out2noise_PSNR: {out2noise_psnr}")
      
      last = self.noisy_images[-1]["image_pt"].detach()
      result = self.model(last)
      result_psnr = self._cal_psnr(self.original_image, result)
      
      if train_history_99["ori_psnr"][-1] > best_psnr:
        best_psnr = train_history_99["ori_psnr"][-1]
        best_epoch = epoch
        best_model_state = self.model.state_dict()
    
        
      
      print("PSNR of reconstruction and original image: ", result_psnr)

     
    e = best_epoch
    shutil.copy2(f'/content/drive/MyDrive/deep_image_prior/output_image/output_{e}_{0}.png', '/content/drive/MyDrive/deep_image_prior/best_picture.png')

    print(
      f"Finish training; best stopping point: Epoch {best_epoch+1}，PSNR: {best_psnr}\n",
      f"Automatically save the model in file {self.out_name}."    
    )

    epoch_count = range(1, len(train_history_99['ori_psnr']) + 1)


    plt.plot(epoch_count, train_history_99['ori_psnr'], 'r-')

    plt.plot(epoch_count, train_history_30['noise_psnr'], 'c--')
    plt.plot(epoch_count, train_history_60['noise_psnr'], 'm--')
    plt.plot(epoch_count, train_history_90['noise_psnr'], 'y--')

    plt.plot(epoch_count, train_history_99['noise_psnr'], 'b--')

    

    plt.legend([f'best: {best_psnr:.3}, epoch: {best_epoch}'])
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.savefig('/content/drive/MyDrive/deep_image_prior/result.png')
    plt.show()

    torch.save(best_model_state, self.out_name)
    

  def _plot_psnr_history(self, ori_psnr, noise_psnr, best_psnr, best_epoch):
    epoch_count = range(1, len(ori_psnr) + 1)
    plt.plot(epoch_count, ori_psnr, 'r-')
    plt.plot(epoch_count, noise_psnr, 'b--')
    plt.legend([f'best: {best_psnr:.3}, epoch: {best_epoch}'])
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.savefig('./result.png')
    plt.show()

      