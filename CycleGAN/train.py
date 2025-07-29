import torch
from dataset import ImgDataSet
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator

def train_fn(disc_I1,disc_I2,gen_I1,gen_I2,loader,opt_disc,opt_gen,L1,mse,d_scaler,g_scaler):
    loop = tqdm(loader,leave=True)
    
    for idx, (img1,img2) in enumerate(loop):
        img1 = img1.to(config.DEVICE)
        img2 = img2.to(config.DEVICE)
        
        # train the discriminators I1 and I2
        with torch.cuda.amp.autocast():  # the autocast converts the f32 to f16
            fake_i1 = gen_I1(img2)
            d_i1_real =  disc_I1(img1)
            d_i1_fake = disc_I1(fake_i1.detach()) # we detach so we use the generator of image in generator part so the gradients shouldnot get mixed up
            i1_reals += d_i1_real.mean().item()
            i1_fakes += d_i1_fake.mean().item()
            d_i1_real_loss = mse(d_i1_real,torch.ones_like(d_i1_real))
            d_i1_fake_loss = mse(d_i1_fake,torch.zeros_like(d_i1_fake))
            d_i1_loss = d_i1_fake_loss + d_i1_real_loss
            
            fake_i2 = gen_I2(img1)
            d_i2_real =  disc_I2(img2)
            d_i2_fake = disc_I2(fake_i2.detach()) # we detach so we use the generator of image in generator part so the gradients shouldnot get mixed up
            d_i2_real_loss = mse(d_i2_real,torch.ones_like(d_i2_real))
            d_i2_fake_loss = mse(d_i2_fake,torch.zeros_like(d_i2_fake))
            d_i2_loss = d_i2_fake_loss + d_i2_real_loss
            
            # put it together
            d_loss = (d_i1_loss + d_i2_loss)/2
            
        opt_disc.zero_grad()
        d_scaler.scale(d_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()
        
        
        # train generators I1 and I2
        with torch.cuda.amp.autocast():
            # adversarial loss
            d_i1_fake = disc_I1(fake_i1)
            d_i2_fake = disc_I2(fake_i2)
            loss_g_i1 = mse(d_i1_fake,torch.ones_like(d_i1_fake))
            loss_g_i2 = mse(d_i2_fake,torch.ones_like(d_i2_fake))
            
            # cycle loss
            cycle_i2 = gen_I2(fake_i1)
            cycle_i1 = gen_I1(fake_i2)
            cycle_i2_loss = L1(img2,cycle_i2)
            cycle_i1_loss = L1(img1,cycle_i1)
            
            #Identity loss
            identity_i2 = gen_I2(img2)
            identity_i1  = gen_I1(img1)      
            identity_i2_loss = L1(img2,identity_i2)
            identity_i1_loss = L1(img1,identity_i1)      
            
            
            # add all together
            g_loss = (
                loss_g_i1 + loss_g_i2 + cycle_i1_loss*config.LAMBDA_CYCLE + cycle_i2_loss*config.LAMBDA_CYCLE + identity_i1_loss*config.LAMBDA_IDENTITY + identity_i2_loss*config.LAMBDA_IDENTITY
            )
            
        opt_gen.zero_grad()
        g_scaler.scale(g_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
        
        if idx % 200 == 0:
            save_image(fake_i1 * 0.5 + 0.5, f"saved_images/i1_{idx}.png")
            save_image(fake_i2 * 0.5 + 0.5, f"saved_images/i2_{idx}.png")

        loop.set_postfix(i1_real=i1_reals / (idx + 1), i1_fake=i1_fakes / (idx + 1))

def main():
    disc_I1 = Discriminator(in_channels=3).to(config.DEVICE)
    disc_I2 = Discriminator(in_channels=3).to(config.DEVICE)
    gen_I1 = Generator(img_channels=3,num_residuals=9).to(config.DEVICE)
    gen_I2 = Generator(img_channels=3,num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_I1.parameters()) + list(disc_I2.parameters()),
        lr = config.LEARNING_RATE,
        betas = (0.5,0.999)
    )
    opt_gen = optim.Adam(
        list(gen_I1.parameters()) + list(gen_I2.parameters()),
        lr = config.LEARNING_RATE,
        betas = (0.5,0.999)
    )
    
    L1 = nn.L1Loss()
    mse = nn.MSELoss()
    
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_I1,gen_I1,opt_gen,config.LEARNING_RATE,
        ) 
        load_checkpoint(
            config.CHECKPOINT_GEN_I2,gen_I2,opt_gen,config.LEARNING_RATE,
        ) 
        load_checkpoint(
            config.CHECKPOINT_CRITIC_I1,disc_I1,opt_disc,config.LEARNING_RATE,
        ) 
        load_checkpoint(
            config.CHECKPOINT_CRITIC_I2,disc_I2,opt_disc,config.LEARNING_RATE,
        ) 
    
    dataset = ImgDataSet(
        root_file1=config.TRAIN_DIR+"",root_file2=config.TRAIN_DIR+"",transform=config.transforms
    )
    loader = DataLoader(
        dataset,
        batch_size= config.BATCH_SIZE,
        shuffle=True,
        num_workers = config.NUM_WORKERS,
        pin_memory=True
    )
    g_scaler = torch.cuda.amp.GradScaler() # to run the operation on f16
    d_scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc_I1,disc_I2,gen_I1,gen_I2,loader,opt_disc,opt_gen,L1,mse,d_scaler,g_scaler)
        
        if config.SAVE_MODEL:
            save_checkpoint(gen_I1,opt_gen,filename=config.CHECKPOINT_GEN_I1)
            save_checkpoint(gen_I2,opt_gen,filename=config.CHECKPOINT_GEN_I2)
            save_checkpoint(disc_I1,opt_disc,filename=config.CHECKPOINT_CRITIC_I1)
            save_checkpoint(disc_I2,opt_disc,filename=config.CHECKPOINT_CRITIC_I2)


if __name__ == "__main__":
    main()