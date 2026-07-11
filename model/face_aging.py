import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from torchvision.utils import save_image
import time
os.makedirs("results", exist_ok=True)
class UTKDataset(Dataset):
    def __init__(self, root):
        self.files = []
        for file in os.listdir(root):
            if(file.endswith(".jpg")):
                self.files.append(os.path.join(root,file))
        
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        
    def age_group(self,age):
        if age <= 10:
            return 0
        elif age <= 20:
            return 1
        elif age<=30:
            return 2
        elif age<=40:
            return 3
        elif age<=50:
            return 4
        elif age<=60:
            return 5
        elif age<=70:
            return 6
        else:
            return 7
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        while True:
            path = self.files[idx]
            try:
                age = int(os.path.basename(path).split("_")[0])
                label = self.age_group(age)
                img = cv2.imread(path)
                if img is None:
                    raise ValueError("Corrupted image")
                img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img  = cv2.resize(img, (64,64))
                img = self.transform(img)
                return img,label
            except Exception:
                idx = (idx + 1)%len(self.files)
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels, affine=True)
        )
    def forward(self, x):
        return x + self.block(x)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), 
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128,256,4,2,1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(256),
            ResidualBlock(256)
        )
    def forward(self,x):
        return self.net(x)
    
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(264, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(True)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(True)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
    def forward(self,z,label):
        label = label.view(-1,8,1,1)
        label = label.repeat(1,1,8,8)
        z = torch.cat([z,label],dim=1)
        x = self.up1(z)
        x = self.up2(x)
        return self.up3(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(11, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), 
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),   
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    def forward(self,x,label):
        label = label.view(-1,8,1,1)
        label = label.repeat(1,1,64,64)
        x = torch.cat([x,label],dim=1)
        return self.net(x)
class AgeClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3,64,4,2,1),
            nn.ReLU(True),
            nn.Conv2d(64,128,4,2,1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128,8)
        )

    def forward(self,x):
        return self.net(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
E = Encoder().to(device)
D = Discriminator().to(device)
G = Generator().to(device)
C = AgeClassifier().to(device)

loss = nn.BCELoss()
l1 = nn.L1Loss()
age_loss = nn.CrossEntropyLoss()
optE = optim.Adam(E.parameters(), lr=0.0002, betas=(0.5, 0.999))
optD = optim.Adam(D.parameters(), lr=0.00005, betas=(0.5, 0.999))
optG = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optC = optim.Adam(C.parameters(), lr=1e-4)
dataset = UTKDataset(
    "/mnt/c/Users/HP/Downloads/UTKFace"
)
# print("Dataset loaded:", len(dataset))
loader = DataLoader(dataset,batch_size=32,shuffle=True,drop_last=True,num_workers=4)
# print("Starting training...")
if __name__ == "__main__":
    for epoch in range(100):
        for i, (images, labels) in enumerate(loader):

            if i % 50 == 0:
                print(f"Batch {i}")
            images = images.to(device)
            labels = labels.to(device)
            target_labels = torch.randint(
            0, 8, (images.size(0),))
            target_idx = target_labels

            target_labels = torch.eye(8)[target_labels]
            target_labels = target_labels.float().to(device)

            pred_age = C(images)
            c_loss = age_loss(pred_age,labels)
            optC.zero_grad()
            c_loss.backward()
            optC.step()

            z = E(images)
            fake = G(z,target_labels)
            real_pred = D(images,torch.eye(8)[labels])
            fake_pred = D(fake.detach(),target_labels)

            #train discriminator
            real_loss = loss(real_pred,torch.ones_like(real_pred))
            fake_loss = loss(fake_pred,torch.zeros_like(fake_pred))
            d_loss = real_loss + fake_loss
            optD.zero_grad()
            d_loss.backward()
            optD.step()
            
            #train generator
            fake = G(z, target_labels)
            fake_pred = D(fake,target_labels)
            adv_loss = loss(fake_pred,torch.ones_like(fake_pred))
            age_pred = C(fake)
            rec_mask = (target_idx == labels).float().view(-1, 1, 1, 1)
            if rec_mask.sum() > 0:
                rec_loss = l1(fake * rec_mask, images * rec_mask) / (rec_mask.mean() + 1e-8)
            else:
                rec_loss = 0.0
            a_loss = age_loss(age_pred,target_idx)
            g_loss = 2.0*adv_loss + 5.0*rec_loss + 1.5*a_loss
            optE.zero_grad()
            optG.zero_grad()
            g_loss.backward()
            optE.step()
            optG.step()
        torch.save(E.state_dict(),"encoder_f.pth")
        torch.save(G.state_dict(),"generator_f.pth")
        torch.save(D.state_dict(),"discriminator_f.pth")
        torch.save(C.state_dict(),"age classifier_f.pth")
        print(
        f"Epoch {epoch+1} | "
        f"D Loss: {d_loss.item():.4f} | "
        f"G Loss: {g_loss.item():.4f}")
        if epoch % 5 == 0:
            with torch.no_grad():
                sample = images[:8]
                z = E(sample)
                target = torch.tensor([7]*8,device=device)
                target_onehot = torch.eye(8,device=device)[target]
                generated = G(z,target_onehot)
                comparison = torch.cat([
                    (sample+1)/2,
                    (generated+1)/2
                ])
                save_image(comparison,f"results/epoch_{epoch}.png",nrow=8)
