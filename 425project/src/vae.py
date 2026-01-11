"""
Variational Autoencoder (VAE) implementations for music feature learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class BasicVAE(nn.Module):
    """
    Basic Variational Autoencoder for music feature extraction.
    Simple fully-connected architecture.
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Optional[list] = None,
        beta: float = 1.0
    ):
        """
        Args:
            input_dim: Dimension of input features
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden layer dimensions (default: [128, 64])
            beta: Beta parameter for beta-VAE (default: 1.0 for standard VAE)
        """
        super(BasicVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space layers
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent space parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation without sampling (use mean)."""
        mu, _ = self.encode(x)
        return mu


def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, 
             logvar: torch.Tensor, beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute VAE loss (reconstruction + KL divergence).
    
    Args:
        recon_x: Reconstructed input
        x: Original input
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Beta parameter for beta-VAE
    
    Returns:
        Total loss, reconstruction loss, KL divergence loss
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


class ConvVAE(nn.Module):
    """
    Convolutional VAE for spectrogram/MFCC features.
    For Medium and Hard tasks.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int],  # (n_mels, time_steps)
        latent_dim: int = 32,
        beta: float = 1.0
    ):
        """
        Args:
            input_shape: Shape of input spectrogram (n_mels, time_steps)
            latent_dim: Dimension of latent space
            beta: Beta parameter for beta-VAE
        """
        super(ConvVAE, self).__init__()
        
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.beta = beta
        
        n_mels, time_steps = input_shape
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # Calculate flattened size after convolutions
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_mels, time_steps)
            dummy_out = self.encoder(dummy)
            self.flattened_size = dummy_out.numel() // dummy_out.size(0)
        
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.flattened_size)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent space parameters."""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        h = self.fc_decode(z)
        h = h.view(h.size(0), 128, self.input_shape[0] // 8, self.input_shape[1] // 8)
        recon = self.decoder(h)
        return recon
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation without sampling (use mean)."""
        mu, _ = self.encode(x)
        return mu


class ConditionalVAE(nn.Module):
    """
    Conditional Variational Autoencoder (CVAE).
    Conditions the VAE on additional information (e.g., genre, language).
    """
    
    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        latent_dim: int = 32,
        hidden_dims: Optional[list] = None,
        beta: float = 1.0
    ):
        """
        Args:
            input_dim: Dimension of input features
            condition_dim: Dimension of condition vector (e.g., one-hot genre encoding)
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden layer dimensions
            beta: Beta parameter for beta-VAE
        """
        super(ConditionalVAE, self).__init__()
        
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.beta = beta
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        # Encoder: input + condition
        encoder_layers = []
        prev_dim = input_dim + condition_dim  # Concatenate input and condition
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space layers
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Decoder: latent + condition
        decoder_layers = []
        prev_dim = latent_dim + condition_dim  # Concatenate latent and condition
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent space parameters (conditioned on condition)."""
        # Concatenate input and condition
        x_cond = torch.cat([x, condition], dim=1)
        h = self.encoder(x_cond)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction (conditioned on condition)."""
        # Concatenate latent and condition
        z_cond = torch.cat([z, condition], dim=1)
        return self.decoder(z_cond)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        mu, logvar = self.encode(x, condition)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, condition)
        return recon, mu, logvar
    
    def get_latent(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Get latent representation without sampling (use mean)."""
        mu, _ = self.encode(x, condition)
        return mu
