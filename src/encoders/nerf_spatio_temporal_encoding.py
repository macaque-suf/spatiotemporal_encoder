import torch
from torch import nn

class NeRFSpatiotemporalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_spatial_distance=100.0, max_time=100):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_spatial_distance = max_spatial_distance
        self.max_time = max_time
        self.scale = 0.5
        
        # Dimensions for intermediate representations
        self.spatial_hidden_dim = 32
        self.pos_embed_dim = 40  # Output dimension for spatial encoding
        self.time_embed_dim = 12  # Output dimension for temporal encoding
        
        # Learned spatial projections - small MLPs for each spatial dimension
        self.spatial_encoder_x = nn.Sequential(
            nn.Linear(1, self.spatial_hidden_dim),
            nn.LayerNorm(self.spatial_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.spatial_hidden_dim, self.spatial_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.spatial_hidden_dim, self.pos_embed_dim // 2)
        )        

        # Add layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.spatial_encoder_y = nn.Sequential(
            nn.Linear(1, self.spatial_hidden_dim),
            nn.LayerNorm(self.spatial_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.spatial_hidden_dim, self.spatial_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.spatial_hidden_dim, self.pos_embed_dim // 2)
        )        
        
        # Number of frequency bands for time
        self.num_time_freqs = 6
        
        # Learnable frequency bands for time
        self.time_freq_bands = nn.Parameter(torch.linspace(0, 5, self.num_time_freqs))
        
        # Separate scaling factors for space and time
        self.spatial_scale = nn.Parameter(torch.ones(1))
        self.temporal_scale = nn.Parameter(torch.ones(1))
        
        # Final projection if needed
        self.final_projection = nn.Linear(self.pos_embed_dim + self.time_embed_dim, embedding_dim)
    
    def forward(self, positions, time_points, cell_features):
        batch_size, num_cells, _ = positions.shape
        device = positions.device
        
        # Normalize inputs
        norm_positions = torch.clamp(positions / self.max_spatial_distance, -1.0, 1.0)
        norm_time = torch.clamp(time_points / self.max_time, 0.0, 1.0)
        
        # Learned spatial encoding
        x_pos = norm_positions[:, :, 0].unsqueeze(-1)
        y_pos = norm_positions[:, :, 1].unsqueeze(-1)
        
        x_encoding = self.spatial_encoder_x(x_pos)
        y_encoding = self.spatial_encoder_y(y_pos)
        
        pos_embed = torch.cat([x_encoding, y_encoding], dim=-1)
        
        # Temporal encoding with learnable frequency bands
        time_embed = torch.zeros(batch_size, num_cells, self.time_embed_dim, device=device)
        time_val = norm_time.squeeze(-1)
        
        for i, freq in enumerate(self.time_freq_bands):
            freq_val = 2.0 ** freq  # Convert to frequency
            # Sin component
            time_embed[:, :, i*2] = torch.sin(freq_val * time_val)
            # Cos component
            time_embed[:, :, i*2 + 1] = torch.cos(freq_val * time_val)
        
        # Apply separate scaling factors
        pos_embed = pos_embed * self.spatial_scale
        time_embed = time_embed * self.temporal_scale
        
        # Combine into single encoding
        spatiotemporal_encoding = torch.cat([pos_embed, time_embed], dim=2)
        
        # Project to match embedding dimension
        spatiotemporal_encoding = self.final_projection(spatiotemporal_encoding)
        
        # Add some dropout to prevent overfitting
        self.dropout = nn.Dropout(0.1)

        # In forward pass
        enhanced_features = cell_features + self.dropout(spatiotemporal_encoding * self.scale)        

        # Add a layer norm for stability
        enhanced_features = self.layer_norm(enhanced_features)
        
        return enhanced_features


