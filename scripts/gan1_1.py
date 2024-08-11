import torch
import torch.nn as nn
import torch.optim as optim
import random

# Generator model
class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 8 * 8 * 1024),
            nn.LeakyReLU(0.2, inplace=True),
            View((-1, 1024, 8, 8)),
            nn.ConvTranspose2d(1024, 512, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(512, eps=1e-5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(256, eps=1e-5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128, eps=1e-5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64, eps=1e-5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(32, eps=1e-5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=5, stride=1, padding=2),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.gen(z)

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * 4 * 4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.disc(x)
        print("Shape after convolutional layers:", x.shape)  # Debugging shape
        x = self.flatten(x)
        print("Shape after flattening:", x.shape)  # Debugging shape
        logits = self.fc(x)
        print("Getting logits",logits)
        out = self.sigmoid(logits)
        print("Getting output",out)
        return out, logits

# Factory function to create a Generator instance
def create_generator(z_dim):
    return Generator(z_dim)

# Factory function to create a Discriminator instance
def create_discriminator():
    return Discriminator()

# Loss function
def model_loss(generator, discriminator, input_real, input_z, device):
    input_real = input_real.to(device)
    input_z = input_z.to(device)
    
    gen_imgs = generator(input_z)
    
    noise_stddev = random.uniform(0.0, 0.1)
    noisy_input_real = input_real + torch.randn_like(input_real) * noise_stddev
    
    d_model_real, d_logits_real = discriminator(noisy_input_real)
    d_model_fake, d_logits_fake = discriminator(gen_imgs)
    
    criterion = nn.BCEWithLogitsLoss()

    real_labels = torch.full(d_model_real.size(), random.uniform(0.9, 1.0), device=device)
    fake_labels = torch.zeros_like(d_model_fake, device=device)

    d_loss_real = criterion(d_logits_real, real_labels)
    d_loss_fake = criterion(d_logits_fake, fake_labels)
    
    d_loss = 0.5 * (d_loss_real + d_loss_fake)
    
    g_loss = criterion(d_logits_fake, torch.ones_like(d_model_fake, device=device))
    
    return d_loss, g_loss

# Training function with epochs
def train(generator, discriminator, dataloader, z_dim, device, num_epochs=200, lr=0.0002):
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)
    generator.to(device)
    discriminator.to(device)
    
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            real_images = data[0].to(device)
            batch_size = real_images.size(0)
            
            z = torch.randn(batch_size, z_dim).to(device)
            
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()
            
            d_loss, g_loss = model_loss(generator, discriminator, real_images, z, device)
            
            d_loss.backward()
            optimizer_D.step()
            
            g_loss.backward()
            optimizer_G.step()
            
            if i % 100 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
        
        torch.save(generator.state_dict(), f"models/generator_epoch_{epoch}.pth")
        torch.save(discriminator.state_dict(), f"models/discriminator_epoch_{epoch}.pth")

# Export functions and classes
__all__ = ["create_generator", "create_discriminator", "model_loss", "train"]

# Version of the file
__version__ = "gan1.1"
