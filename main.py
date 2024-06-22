import torch
import json
from src.trainer.Only_DIP_trainer import Only_DIP_Trainer
from src.trainer.Hier_DIP_trainer import Hier_DIP_Trainer
from src.model.skip import get_net
from src.noisy_image import Noisy_Image_Generator

def main():
    config = None
    with open("config.json") as file:
        config = json.load(file)

    root_dir = config["root_directory"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.cuda.FloatTensor

    noisy_image_generator = Noisy_Image_Generator(
        input_image_path = root_dir + config["image_without_noise"],
        noise_scale = 25/255,
        num = config["num_noisy_images"],
        device = device
    )

    original_image, prev_noisy_images, noisy_images = noisy_image_generator.generate(
        output_dir = root_dir + "noise_image"
    )
    
    net = get_net(
        input_depth = 3,
        pad = 'reflection',
        skip_n33d=128, 
        skip_n33u=128, 
        skip_n11=4, 
        num_scales=5,
        upsample_mode='bilinear'
    ).type(dtype)

    
    loss = torch.nn.MSELoss().type(dtype)
    optimizer = torch.optim.Adam(net.parameters(), lr=config["learning_rate"])

    trainer = None
    if config["model_type"] == "only_DIP":
        trainer = Only_DIP_Trainer(
            num_epochs = config["num_epochs"],
            noisy_images = noisy_images,
            prev_noisy_images = prev_noisy_images,
            model = net,
            original_image = original_image,
            criterion = loss,
            optimizer = optimizer,
            device = device,
            out_name = root_dir + config["output_model_name"]
        )

    elif config["model_type"] == "hier_DIP":
        trainer = Hier_DIP_Trainer(
            num_epochs = config["num_epochs"],
            noisy_images = noisy_images,
            prev_noisy_images = prev_noisy_images,
            model = net,
            original_image = original_image,
            criterion = loss,
            optimizer = optimizer,
            device = device,
            out_name = root_dir + config["output_model_name"]
        )


    trainer.train()



if __name__ == "__main__":
    print(f"Able to use cuda ? {torch.cuda.is_available()}")
    main()