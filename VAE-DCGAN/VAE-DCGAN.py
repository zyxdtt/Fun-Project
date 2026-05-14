import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, ConcatDataset

class Encoder(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(256*3*3, 512), nn.ReLU()
        )
        self.mu = nn.Linear(512, z_dim)
        self.logvar = nn.Linear(512, z_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

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
test  = datasets.MNIST('./data', train=False, download=True, transform=transform)
train_loader = DataLoader(ConcatDataset([train, test]), batch_size=256, shuffle=True)

E = Encoder(100).to(device)
G = Generator(100).to(device)
D = Discriminator().to(device)

opt_E = optim.Adam(E.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

bce_loss = nn.BCELoss()
mse_loss = nn.MSELoss()   

lambda_rec = 1.0    
lambda_kl  = 0.05    

for epoch in range(1, 21):
    for real, _ in train_loader:
        real = real.to(device)
        bs = real.size(0)
        real_label = torch.ones(bs, 1).to(device)
        fake_label = torch.zeros(bs, 1).to(device)
        out_real = D(real)
        loss_D_real = bce_loss(out_real, real_label)
        z_rand = torch.randn(bs, 100).to(device)
        fake_rand = G(z_rand).detach()
        loss_D_fake_rand = bce_loss(D(fake_rand), fake_label)
        with torch.no_grad():
            mu, logvar = E(real)
            z_rec = E.reparameterize(mu, logvar)
            fake_rec = G(z_rec).detach()
        loss_D_fake_rec = bce_loss(D(fake_rec), fake_label)
        loss_D = loss_D_real + loss_D_fake_rand + loss_D_fake_rec
        opt_D.zero_grad(); loss_D.backward(); opt_D.step()
        mu, logvar = E(real)
        z_rec = E.reparameterize(mu, logvar)
        fake_rec = G(z_rec)       
        z_rand = torch.randn(bs, 100).to(device)
        fake_rand = G(z_rand)
        loss_G_adv = bce_loss(D(fake_rand), real_label) + bce_loss(D(fake_rec), real_label)
        loss_rec = mse_loss(fake_rec, real)
        loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss_EG = loss_G_adv + lambda_rec * loss_rec + lambda_kl * loss_kl
        opt_E.zero_grad(); opt_G.zero_grad()
        loss_EG.backward()
        opt_E.step(); opt_G.step()

    print(f"Epoch {epoch} | D loss: {loss_D.item():.3f} | EG loss: {loss_EG.item():.3f}")

with torch.no_grad():
    z = torch.randn(64, 100).to(device)
    fake_samples = G(z).cpu()
    save_image(fake_samples, "dcgan_vae_samples.png", nrow=8, normalize=True)
print("Samples saved to dcgan_vae_samples.png")