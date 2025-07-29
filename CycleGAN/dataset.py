from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class ImgDataSet(Dataset):
    def __init__(self,root_file1,root_file2,transform=None):
        super().__init__()
        self.root_file1 = root_file1
        self.root_file2 = root_file2
        self.transform = transform
        
        self.imgs_1 = os.listdir(self.root_file1)
        self.imgs_2 = os.listdir(self.root_file2)
        self.length_dataset = max(len(self.imgs_1,self.imgs_2))
        
        self.img_1_len = len(self.imgs_1)
        self.img_2_len = len(self.imgs_2)
        
    def __len__(self):
        return self.length_dataset
    
    def __getitem__(self,index):
        img1 = self.imgs_1[index % self.img_1_len]
        img2 = self.imgs_2[index % self.img_2_len]
        
        img1_path = os.path.join(self.root_file1,img1)
        img2_path = os.path.join(self.root_file2,img2)
        
        img1 = np.array(Image.open(img1_path).convert("RGB"))
        img2 = np.array(Image.open(img2_path).convert("RGB"))
        
        if self.transform:
            augumentations = self.transform(image=img1,image0=img2)
            img1 = augumentations["image"]
            img2 = augumentations["image0"]
            
        return img1,img2