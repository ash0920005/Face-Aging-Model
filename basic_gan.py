from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import os
import torch

batch_size = 128
lr = 0.0002
latent_dim = 100
epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

loader = DataLoader(dataset, batch_size=128, shuffle=True)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,784),
            nn.Tanh()
        )
    def forward(self,x):
        x = self.model(x)
        return x.view(-1,1,28,28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784,512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x = x.view(x.size(0),-1)
        return self.model(x)
    
G = Generator()
D = Discriminator()
criterion = nn.BCELoss()
g_optimizer = optim.Adam(
    G.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)

d_optimizer = optim.Adam(
    D.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)

os.makedirs("generated", exist_ok=True)
fixed_noise = torch.randn(64, latent_dim).to(device)

for epoch in range(epochs):
    total_d_loss = 0
    total_g_loss = 0
    num_batches = 0
    for real_images, _  in loader:
        real_images = real_images.to(device)
        batch_size_curr = real_images.size(0)
        real_labels = torch.ones(batch_size_curr, 1, device=device)
        fake_labels = torch.zeros(batch_size_curr,1,device=device)
        z = torch.randn(batch_size_curr,latent_dim,device=device)
        fake_imgs = G(z)
        real_output = D(real_images)
        fake_output = D(fake_imgs.detach())
        real_loss = criterion(real_output,real_labels)
        fake_loss = criterion(fake_output,fake_labels)
        d_loss = real_loss + fake_loss
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        g_loss = fake_loss

        z = torch.randn(batch_size_curr,latent_dim,device=device)
        fake_imgs = G(z)
        output = D(fake_imgs)
        g_loss = criterion(output,real_labels)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        total_d_loss += d_loss.item()
        total_g_loss += g_loss.item()
        num_batches += 1
        
        with torch.no_grad():
            samples = G(fixed_noise)

        save_image(
            samples,
            f"generated/epoch_{epoch+1}.png",
            nrow=8,
            normalize=True
        )
    avg_d_loss = total_d_loss / num_batches
    avg_g_loss = total_g_loss / num_batches

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"D Loss: {avg_d_loss:.4f} "
        f"G Loss: {avg_g_loss:.4f}"
    )