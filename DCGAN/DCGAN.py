import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader,ConcatDataset

class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256*7*7),
            nn.BatchNorm1d(256*7*7), nn.ReLU(),
            nn.Unflatten(1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 3, 1, 1), nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(256*3*3, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.ToTensor()
train = datasets.MNIST('./data', train=True, download=True, transform=transform)
test=datasets.MNIST('./data',train=False,download=True,transform=transform)
train_loader=DataLoader(ConcatDataset(train,test),batch_size=256,shuffle=True)

G = Generator(100).to(device)
D = Discriminator().to(device)
opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
loss_fn = nn.BCELoss()

for epoch in range(1, 21):
    for real, _ in train_loader:
        real = real.to(device)
        batch_size = real.size(0)
        real_label = torch.ones(batch_size, 1).to(device)
        fake_label = torch.zeros(batch_size, 1).to(device)
        out_real = D(real)
        loss_D_real = loss_fn(out_real, real_label)
        z = torch.randn(batch_size, 100).to(device)
        fake = G(z).detach()  
        out_fake = D(fake)
        loss_D_fake = loss_fn(out_fake, fake_label)
        loss_D = loss_D_real + loss_D_fake
        opt_D.zero_grad(); loss_D.backward(); opt_D.step()
        z = torch.randn(batch_size, 100).to(device)
        fake = G(z)
        out = D(fake)
        loss_G = loss_fn(out, real_label)  
        opt_G.zero_grad(); loss_G.backward(); opt_G.step()
    print(f"Epoch {epoch} done")

with torch.no_grad():
    z = torch.randn(64, 100).to(device)
    fake_samples = G(z).cpu()
    save_image(fake_samples, "dcgan_samples.png", nrow=8, normalize=True)
print("Samples saved to dcgan_samples.png")