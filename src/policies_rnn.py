
import torch
import torch.nn as nn
from policies import VariationalEncoder

class RecurrentVariationalEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        # x is (B, Seq, InputDim)
        # We want to output (mu, logvar) for *each* time step or just the final?
        # For hedging, we need action at each time step t.
        # So we return sequence of latents: (B, Seq, LatentDim)
        
        output, (hn, cn) = self.lstm(x)
        # output: (B, Seq, HiddenDim)
        
        mu = self.mu(output)
        logvar = self.logvar(output)
        
        # Soft-clamp
        logvar = torch.clamp(logvar, min=-10.0, max=2.0)
        
        return mu, logvar

class RecurrentFactorizedVariationalPolicy(nn.Module):
    """
    LSTM-based VIB Policy.
    Encodes history x_{0:t} -> z_t -> action_t
    """
    def __init__(self, input_dim, latent_dim_per_feature=1, hidden_dim=32):
        super().__init__()
        self.input_dim = input_dim
        
        # Factorized or Monolithic? 
        # State paper used factorized per feature.
        # Doing element-wise LSTM is overkill and loses cross-feature correlation.
        # Let's do a Monolithic LSTM encoder for the "Micro" channel, 
        # and a simple MLP (or Identity) for the "Greeks" channel?
        # OR: Just one big LSTM VIB.
        
        # To strictly comparable with FactorizedVariationalPolicy:
        # We can treat the LSTM strictly as the "Micro Encoder" replacement.
        # But 'combined' features enter together.
        
        # Simpler approach: One LSTM Encoder taking all features.
        self.encoder = RecurrentVariationalEncoder(
            input_dim, 
            latent_dim=input_dim * latent_dim_per_feature, # Same total latent cap
            hidden_dim=hidden_dim
        )
        
        total_latent = input_dim * latent_dim_per_feature
        
        self.decoder = nn.Sequential(
            nn.Linear(total_latent, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        # x: (B, Seq, D) or (B, D)
        # If (B, D), we interpret as Seq=1 (stateless use) or need hidden state?
        # This is the tricky part of training interface.
        # If we pass (B, Seq, D), we perform full sequence hedging.
        
        if x.dim() == 2:
            x = x.unsqueeze(1) # (B, 1, D)
            
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        
        # Decode action
        action = self.decoder(z)
        
        return action.squeeze(-1), [mu], [logvar] # Return as lists to match API
