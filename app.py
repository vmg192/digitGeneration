# app.py (Updated Version)

import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# --- Re-define the Model Architecture ---
# This MUST be the same architecture as used in training.
IMG_SIZE = 28
LATENT_DIM = 10
N_CLASSES = 10

class ConditionalVAE(nn.Module):
    def __init__(self):
        super(ConditionalVAE, self).__init__()
        input_dim = IMG_SIZE * IMG_SIZE + N_CLASSES
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, LATENT_DIM)
        self.fc_log_var = nn.Linear(256, LATENT_DIM)
        decoder_input_dim = LATENT_DIM + N_CLASSES
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, IMG_SIZE * IMG_SIZE), nn.Tanh()
        )

    def encode(self, x, y):
        y_onehot = nn.functional.one_hot(y, N_CLASSES).float()
        combined = torch.cat([x.view(x.size(0), -1), y_onehot], dim=1)
        h = self.encoder(combined)
        return self.fc_mu(h), self.fc_log_var(h)
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        y_onehot = nn.functional.one_hot(y, N_CLASSES).float()
        combined = torch.cat([z, y_onehot], dim=1)
        return self.decoder(combined).view(-1, 1, IMG_SIZE, IMG_SIZE)
        
    def forward(self, x, y):
        mu, log_var = self.encode(x, y)
        z = self.reparameterize(mu, log_var)
        return self.decode(z, y), mu, log_var

# --- Load the Pre-Trained Model ---
@st.cache_resource
def load_model():
    device = torch.device('cpu')
    model = ConditionalVAE().to(device)
    model.load_state_dict(torch.load('cvae_mnist.pth', map_location=device))
    model.eval()
    return model

model = load_model()

# --- Web App User Interface ---
st.set_page_config(layout="wide")
st.title("Handwritten Digit Generation using a cVAE")

st.markdown("---") # Adds a horizontal line

# --- Controls on the main page ---
st.subheader("Controls")
col1, col2 = st.columns([1, 3]) # Create two columns for layout

with col1:
    selected_digit = st.selectbox("Select a digit to generate:", list(range(10)))

with col2:
    # Add some vertical space to align the button better
    st.write("") 
    st.write("")
    generate_button = st.button("Generate Images", type="primary")

st.markdown("---")

# --- Generation and Display Logic ---
if generate_button:
    st.subheader(f"Generating 5 images for the digit: {selected_digit}")

    # Generate images
    with torch.no_grad():
        z = torch.randn(5, LATENT_DIM)
        label = torch.LongTensor([selected_digit] * 5)
        generated_images_tensor = model.decode(z, label)
        generated_images = (generated_images_tensor * 0.5 + 0.5).numpy()

    # Display images in 5 columns
    cols = st.columns(5)
    for i, col in enumerate(cols):
        with col:
            st.image(generated_images[i].squeeze(), caption=f"Image {i+1}", width=128)
else:
    st.info("Select a digit and click 'Generate Images' to start.")
