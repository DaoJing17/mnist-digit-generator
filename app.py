import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Define the same CVAE class
class CVAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.embed = nn.Embedding(10, 10)

        self.encoder = nn.Sequential(
            nn.Linear(28*28 + 10, 400),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)

        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim + 10, 400),
            nn.ReLU(),
            nn.Linear(400, 28*28),
            nn.Sigmoid()
        )

    def encode(self, x, labels):
        x = x.view(x.size(0), -1)
        labels = self.embed(labels)
        x = torch.cat([x, labels], dim=1)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        labels = self.embed(labels)
        z = torch.cat([z, labels], dim=1)
        return self.decoder_fc(z).view(-1, 1, 28, 28)

    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CVAE().to(device)
model.load_state_dict(torch.load("cvae_mnist.pth", map_location=device))
model.eval()

# Streamlit UI
st.title("Handwritten Digit Generator (0–9)")
digit = st.selectbox("Choose a digit to generate:", list(range(10)))
num_samples = 5

if st.button("Generate Images"):
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim).to(device)
        labels = torch.tensor([digit] * num_samples).to(device)
        samples = model.decode(z, labels).cpu().numpy()

        # Display images
        st.write("### Generated Images:")
        cols = st.columns(num_samples)
        for i in range(num_samples):
            image = samples[i][0]
            cols[i].image(image, width=100, clamp=True, channels="GRAY")
