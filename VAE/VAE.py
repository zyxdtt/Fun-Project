import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader,ConcatDataset

class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x.reshape(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
mnist_test=datasets.MNIST('./data',train=False,download=True,transform=transforms.ToTensor())
train_loader=DataLoader(ConcatDataset([mnist_train, mnist_test]),batch_size=128,shuffle=True)

net = VAE(latent_dim=20).to(device)
optimizer = optim.Adam(net.parameters(), lr=1e-3)

for epoch in range(1, 11):
    net.train()
    total_loss = 0
    for data, _ in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = net(data)
        l = loss_fn(recon, data, mu, logvar)
        l.backward()
        total_loss += l.item()
        optimizer.step()
    print(f'Epoch {epoch}, Avg Loss: {total_loss / len(train_loader.dataset):.4f}')

with torch.no_grad():
    z = torch.randn(64, 20).to(device)
    samples = net.decode(z).reshape(-1, 1, 28, 28)
    save_image(samples, 'vae_samples.png', nrow=8, normalize=True)
print("Samples saved to vae_samples.png")