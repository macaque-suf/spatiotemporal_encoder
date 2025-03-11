import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

from encoders.nerf_spatio_temporal_encoding import NeRFSpatiotemporalEncoding

class MorphogenesisDataGenerator:
    """
    Generates synthetic data for spatiotemporal encoding based on natural morphogenesis algorithms.
    """
    def __init__(self, 
                 spatial_size=100, 
                 max_time=100, 
                 feature_dim=64, 
                 num_morphogens=4,
                 diffusion_rates=None,
                 decay_rates=None):
        """
        Initialize the data generator.
        
        Args:
            spatial_size: Size of the spatial domain
            max_time: Maximum simulation time
            feature_dim: Dimension of cell features
            num_morphogens: Number of morphogen gradients to simulate
            diffusion_rates: Diffusion rates for morphogens (default: random values)
            decay_rates: Decay rates for morphogens (default: random values)
        """
        self.spatial_size = spatial_size
        self.max_time = max_time
        self.feature_dim = feature_dim
        self.num_morphogens = num_morphogens
        
        # Initialize diffusion and decay rates if not provided
        self.diffusion_rates = diffusion_rates if diffusion_rates is not None else \
                              np.random.uniform(0.1, 1.0, num_morphogens)
        self.decay_rates = decay_rates if decay_rates is not None else \
                          np.random.uniform(0.01, 0.1, num_morphogens)
                          
        # Precompute some values for reaction-diffusion
        self.dt = 1.0  # Time step for simulation
        
    def reaction_diffusion(self, num_cells=100, time_steps=20, seed=None):
        """
        Generate cell positions and features using a reaction-diffusion system 
        (inspired by Turing patterns).
        
        Returns:
            positions: Tensor of cell positions [batch_size, num_cells, 2]
            time_points: Tensor of time points [batch_size, num_cells, 1]
            cell_features: Tensor of cell features [batch_size, num_cells, feature_dim]
            targets: Target features [batch_size, num_cells, feature_dim]
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
        # Initialize concentration fields (U and V in reaction-diffusion)
        grid_size = int(np.sqrt(self.spatial_size))
        u = np.random.random((grid_size, grid_size))
        v = np.random.random((grid_size, grid_size))
        
        # Parameters for Gray-Scott model
        Du, Dv = 0.16, 0.08  # Diffusion rates
        f, k = 0.035, 0.065  # Feed, kill rates
        
        # Store states over time for visualization and sampling
        states = []
        
        # Run reaction-diffusion simulation
        for t in range(time_steps):
            # Compute Laplacians
            laplacian_u = self._laplacian(u)
            laplacian_v = self._laplacian(v)
            
            # Update concentrations
            uvv = u * v * v
            u_next = u + Du * laplacian_u - uvv + f * (1 - u)
            v_next = v + Dv * laplacian_v + uvv - (f + k) * v
            
            # Clip to avoid instabilities
            u = np.clip(u_next, 0, 1)
            v = np.clip(v_next, 0, 1)
            
            # Store state
            combined_state = np.stack([u, v], axis=0)  # [2, grid_size, grid_size]
            states.append(combined_state)
        
        # Convert to tensor for easier manipulation
        states = np.stack(states, axis=0)  # [time_steps, 2, grid_size, grid_size]
        
        # Sample random cells based on concentration gradients
        positions = []
        time_points = []
        
        # Use v-concentration as probability field for cell positions
        # Higher v values more likely to have cells
        # Loop through time steps and sample cell positions
        for t in range(time_steps):
            # Normalize v to create a probability distribution
            prob_field = states[t, 1]  # v concentration
            prob_field = prob_field / np.sum(prob_field)
            
            # Sample indices
            idx = np.random.choice(grid_size * grid_size, 
                                   size=num_cells // time_steps, 
                                   p=prob_field.flatten(),
                                   replace=True)
            
            # Convert to 2D positions
            y_idx, x_idx = np.unravel_index(idx, (grid_size, grid_size))
            
            # Add noise to positions
            x_pos = x_idx + np.random.normal(0, 0.5, len(x_idx))
            y_pos = y_idx + np.random.normal(0, 0.5, len(y_idx))
            
            # Scale to spatial size
            x_pos = (x_pos / grid_size) * self.spatial_size
            y_pos = (y_pos / grid_size) * self.spatial_size
            
            # Combine positions
            pos = np.stack([x_pos, y_pos], axis=1)
            positions.append(pos)
            
            # Create time points
            t_scaled = (t / time_steps) * self.max_time
            time_points.append(np.full((len(x_pos), 1), t_scaled))
        
        # Concatenate all positions and time points
        positions = np.vstack(positions)
        time_points = np.vstack(time_points)
        
        # If we didn't get exactly num_cells, adjust by sampling
        if len(positions) != num_cells:
            indices = np.random.choice(len(positions), num_cells, replace=len(positions) < num_cells)
            positions = positions[indices]
            time_points = time_points[indices]
        
        # Generate cell features based on morphogen gradients
        cell_features = self._generate_cell_features(positions, time_points)
        
        # Generate targets with temporal dependency
        targets = self._generate_targets(positions, time_points, cell_features)
        
        # Convert to tensors and reshape
        positions_tensor = torch.tensor(positions, dtype=torch.float32).unsqueeze(0)
        time_points_tensor = torch.tensor(time_points, dtype=torch.float32).unsqueeze(0)
        cell_features_tensor = torch.tensor(cell_features, dtype=torch.float32).unsqueeze(0)
        targets_tensor = torch.tensor(targets, dtype=torch.float32).unsqueeze(0)
        
        return positions_tensor, time_points_tensor, cell_features_tensor, targets_tensor
        
    def cell_division_model(self, num_cells=100, time_steps=20, division_prob=0.1, seed=None):
        """
        Generate data based on a cell division and movement model.
        
        Args:
            num_cells: Target number of cells
            time_steps: Number of time steps to simulate
            division_prob: Probability of cell division per time step
            seed: Random seed for reproducibility
            
        Returns:
            positions: Tensor of cell positions [batch_size, num_cells, 2]
            time_points: Tensor of time points [batch_size, num_cells, 1]
            cell_features: Tensor of cell features [batch_size, num_cells, feature_dim]
            targets: Target features [batch_size, num_cells, feature_dim]
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
        # Start with a few cells
        init_num_cells = max(5, num_cells // 10)
        
        # Initialize cell positions randomly
        cell_positions = np.random.uniform(0, self.spatial_size, (init_num_cells, 2))
        
        # Cell birth times and parent indices
        birth_times = np.zeros(init_num_cells)
        parent_indices = np.zeros(init_num_cells, dtype=int)
        
        # Store states over time
        all_positions = []
        all_times = []
        all_parents = []
        
        # Run simulation
        current_time = 0
        
        for t in range(time_steps):
            current_time = (t / time_steps) * self.max_time
            
            # Cell movement - random walk with neighbor interactions
            if len(cell_positions) > 1:
                # Calculate distances between cells
                distances = squareform(pdist(cell_positions))
                np.fill_diagonal(distances, np.inf)  # Ignore self-distances
                
                # Repulsion/attraction forces
                forces = np.zeros_like(cell_positions)
                
                for i in range(len(cell_positions)):
                    # Get nearest neighbors
                    nearest_idx = np.argsort(distances[i])[:5]  # Consider 5 nearest neighbors
                    
                    for j in nearest_idx:
                        # Vector from j to i
                        vec = cell_positions[i] - cell_positions[j]
                        dist = np.linalg.norm(vec)
                        
                        # Skip if too far
                        if dist > 10:
                            continue
                            
                        # Normalize
                        vec = vec / (dist + 1e-8)
                        
                        # Repulsion if too close, weak attraction if just right
                        if dist < 5:
                            # Strong repulsion when too close
                            force = vec * (1.0 / (dist + 1e-3)) * 2.0
                        else:
                            # Weak attraction at medium distance
                            force = -vec * 0.05
                            
                        forces[i] += force
                
                # Apply forces with some friction and noise
                cell_positions += forces * 0.5
                
            # Random movement
            cell_positions += np.random.normal(0, 1.0, cell_positions.shape)
            
            # Keep within bounds
            cell_positions = np.clip(cell_positions, 0, self.spatial_size)
            
            # Cell division
            new_cells = []
            new_times = []
            new_parents = []
            
            for i in range(len(cell_positions)):
                # Probability of division depends on cell age
                cell_age = current_time - birth_times[i]
                div_probability = division_prob * (1 - np.exp(-cell_age / 20))
                
                if np.random.random() < div_probability:
                    # Create daughter cell with small displacement
                    displacement = np.random.normal(0, 2.0, 2)
                    daughter_pos = cell_positions[i] + displacement
                    
                    # Ensure it's within bounds
                    daughter_pos = np.clip(daughter_pos, 0, self.spatial_size)
                    
                    new_cells.append(daughter_pos)
                    new_times.append(current_time)
                    new_parents.append(i)
            
            # Add new cells
            if new_cells:
                cell_positions = np.vstack([cell_positions, np.array(new_cells)])
                birth_times = np.append(birth_times, np.array(new_times))
                parent_indices = np.append(parent_indices, np.array(new_parents))
            
            # Store current state
            all_positions.append(cell_positions.copy())
            all_times.append(np.full(len(cell_positions), current_time))
            all_parents.append(parent_indices.copy())
            
            # Break if we exceed the target number of cells
            if len(cell_positions) >= num_cells * 2:
                break
        
        # Sample evenly across time to get our target number of cells
        all_positions_flat = np.vstack([pos for pos in all_positions])
        all_times_flat = np.concatenate([times for times in all_times])
        
        # If we have too many cells, sample randomly
        if len(all_positions_flat) > num_cells:
            indices = np.random.choice(len(all_positions_flat), num_cells, replace=False)
            positions = all_positions_flat[indices]
            time_points = all_times_flat[indices].reshape(-1, 1)
        else:
            # If not enough cells, duplicate some
            indices = np.random.choice(len(all_positions_flat), num_cells, replace=True)
            positions = all_positions_flat[indices]
            time_points = all_times_flat[indices].reshape(-1, 1)
        
        # Generate cell features based on morphogen gradients
        cell_features = self._generate_cell_features(positions, time_points)
        
        # Generate targets with temporal dependency
        targets = self._generate_targets(positions, time_points, cell_features)
        
        # Convert to tensors and reshape
        positions_tensor = torch.tensor(positions, dtype=torch.float32).unsqueeze(0)
        time_points_tensor = torch.tensor(time_points, dtype=torch.float32).unsqueeze(0)
        cell_features_tensor = torch.tensor(cell_features, dtype=torch.float32).unsqueeze(0)
        targets_tensor = torch.tensor(targets, dtype=torch.float32).unsqueeze(0)
        
        return positions_tensor, time_points_tensor, cell_features_tensor, targets_tensor
    
    def generate_batch(self, batch_size=32, num_cells=100, method='mixed', **kwargs):
        """
        Generate a batch of data, potentially using a mix of methods.
        
        Args:
            batch_size: Number of samples in the batch
            num_cells: Number of cells per sample
            method: One of 'reaction_diffusion', 'cell_division', or 'mixed'
            
        Returns:
            batch_data: Tuple of (positions, time_points, cell_features, targets)
        """
        all_positions = []
        all_time_points = []
        all_features = []
        all_targets = []

        pos, time, feat, targ = None, None, None, None
        
        for _ in range(batch_size):
            # Choose method
            if method == 'mixed':
                chosen_method = np.random.choice(['reaction_diffusion', 'cell_division'])
            else:
                chosen_method = method
            
            # Generate data
            seed = np.random.randint(0, 10000)
            if chosen_method == 'reaction_diffusion':
                pos, time, feat, targ = self.reaction_diffusion(
                    num_cells=num_cells, 
                    seed=seed,
                    **kwargs
                )
            elif chosen_method == 'cell_division':
                pos, time, feat, targ = self.cell_division_model(
                    num_cells=num_cells, 
                    seed=seed,
                    **kwargs
                )

            all_positions.append(pos)
            all_time_points.append(time)
            all_features.append(feat)
            all_targets.append(targ)
        
        # Concatenate along batch dimension
        batch_positions = torch.cat(all_positions, dim=0)
        batch_time_points = torch.cat(all_time_points, dim=0)
        batch_features = torch.cat(all_features, dim=0)
        batch_targets = torch.cat(all_targets, dim=0)
        
        return batch_positions, batch_time_points, batch_features, batch_targets
    
    def _generate_cell_features(self, positions, time_points):
        """
        Generate cell features based on diffusing morphogen gradients.
        
        Args:
            positions: Array of cell positions [num_cells, 2]
            time_points: Array of time points [num_cells, 1]
            
        Returns:
            features: Cell features [num_cells, feature_dim]
        """
        num_cells = len(positions)
        
        # Initialize features with random noise
        features = np.random.normal(0, 0.1, (num_cells, self.feature_dim))
        
        # Create morphogen sources
        num_sources = 3 + int(np.random.randint(2, 5))
        source_positions = np.random.uniform(0, self.spatial_size, (num_sources, 2))
        source_strengths = np.random.uniform(0.5, 2.0, num_sources)
        source_start_times = np.random.uniform(0, self.max_time/2, num_sources)
        
        # Create spatial masks for feature patterns
        # Each morphogen affects different feature dimensions
        pattern_indices = []
        remaining_dims = list(range(self.feature_dim))
        
        for i in range(self.num_morphogens):
            # Select random subset of feature dimensions for this morphogen
            num_dims = max(1, int(self.feature_dim / self.num_morphogens * 
                                  np.random.uniform(0.5, 1.5)))
            num_dims = min(num_dims, len(remaining_dims))
            
            if num_dims == 0:
                pattern_indices.append([])
                continue
                
            # Sample without replacement
            selected_dims = np.random.choice(remaining_dims, num_dims, replace=False)
            pattern_indices.append(selected_dims)
            
            # Remove selected dimensions
            for dim in selected_dims:
                remaining_dims.remove(dim)
        
        # Apply remaining dimensions to random patterns
        if remaining_dims:
            for dim in remaining_dims:
                # Add to a random pattern
                idx = np.random.randint(0, self.num_morphogens)
                pattern_indices[idx] = np.append(pattern_indices[idx], dim)
        
        # For each morphogen source, compute its effect on cell features
        for s in range(num_sources):
            src_pos = source_positions[s]
            strength = source_strengths[s]
            start_time = source_start_times[s]
            
            # Select which morphogen gradient this source produces
            morphogen_idx = np.random.randint(0, self.num_morphogens)
            diffusion_rate = self.diffusion_rates[morphogen_idx]
            decay_rate = self.decay_rates[morphogen_idx]
            
            # Calculate squared distances from this source to all cells
            distances = np.sum((positions - src_pos) ** 2, axis=1)
            
            # Calculate effect based on distance and time
            # Apply diffusion equation: C = (A/(4πDt)) * exp(-r²/(4Dt)) * exp(-kt)
            for i in range(num_cells):
                time = time_points[i, 0] - start_time
                
                # Skip if before start time
                if time <= 0:
                    continue
                    
                # Diffusion and decay
                diff_factor = 4 * np.pi * diffusion_rate * time
                if diff_factor <= 0:
                    continue
                    
                dist = distances[i]
                concentration = (strength / diff_factor) * \
                                np.exp(-dist / (4 * diffusion_rate * time)) * \
                                np.exp(-decay_rate * time)
                
                # Apply to selected feature dimensions with some nonlinearity
                if len(pattern_indices[morphogen_idx]) > 0:
                    features[i, pattern_indices[morphogen_idx]] += \
                        concentration * np.random.uniform(0.8, 1.2, len(pattern_indices[morphogen_idx]))
        
        # Apply some nonlinear transformations to make it more interesting
        # Simulate gene regulation networks
        for i in range(2):  # Apply multiple layers of regulation
            # Random linear combination
            weights = np.random.normal(0, 1.0, (self.feature_dim, self.feature_dim))
            bias = np.random.normal(0, 0.1, self.feature_dim)
            
            # Apply with nonlinearity (sigmoid-like)
            features = np.tanh(features @ weights + bias)
        
        return features
        
    def _generate_targets(self, positions, time_points, features):
        """
        Generate target values based on positions, time, and features.
        This simulates the expected biological outcome.
        
        Args:
            positions: Array of cell positions [num_cells, 2]
            time_points: Array of time points [num_cells, 1]
            features: Cell features [num_cells, feature_dim]
            
        Returns:
            targets: Target features [num_cells, feature_dim]
        """
        num_cells = len(positions)
        
        # Start with features as baseline
        targets = features.copy()
        
        # Simulate a gene regulatory network effect
        # (neighbors influence each other)
        
        # Build KD-tree for fast nearest neighbor lookup
        from scipy.spatial import KDTree
        tree = KDTree(positions)
        
        # Find neighbors within a certain radius
        neighbors = tree.query_ball_point(positions, r=10.0)
        
        # Apply neighbor influence
        for i in range(num_cells):
            if len(neighbors[i]) <= 1:  # Skip if no neighbors (except self)
                continue
                
            # Average neighbor features (exclude self)
            neighbor_idx = [n for n in neighbors[i] if n != i]
            if not neighbor_idx:
                continue
                
            neighbor_features = features[neighbor_idx]
            avg_neighbor = np.mean(neighbor_features, axis=0)
            
            # Apply neighbor influence with time-dependent factor
            # (stronger influence at later times)
            time_factor = min(1.0, time_points[i, 0] / (self.max_time * 0.5))
            influence = 0.2 * time_factor
            
            targets[i] = (1 - influence) * targets[i] + influence * avg_neighbor
        
        # Add temporal progression effects
        time_normalized = time_points / self.max_time
        
        # Different feature dimensions have different temporal dynamics
        for d in range(self.feature_dim):
            # Randomly choose dynamic type
            dynamic_type = np.random.choice(['linear', 'exponential', 'sigmoid', 'oscillatory'])
            
            if dynamic_type == 'linear':
                # Linear change over time
                rate = np.random.uniform(-0.5, 0.5)
                targets[:, d] += rate * time_normalized.flatten()
                
            elif dynamic_type == 'exponential':
                # Exponential growth or decay
                rate = np.random.uniform(-0.05, 0.05)
                targets[:, d] *= np.exp(rate * 10 * time_normalized.flatten())
                
            elif dynamic_type == 'sigmoid':
                # Sigmoidal switch
                midpoint = np.random.uniform(0.3, 0.7)
                steepness = np.random.uniform(5, 15)
                shift = 1.0 / (1.0 + np.exp(-steepness * (time_normalized.flatten() - midpoint)))
                magnitude = np.random.uniform(0.1, 0.5)
                targets[:, d] += magnitude * shift
                
            elif dynamic_type == 'oscillatory':
                # Oscillatory pattern
                frequency = np.random.uniform(1, 5) 
                amplitude = np.random.uniform(0.05, 0.2)
                phase = np.random.uniform(0, 2*np.pi)
                targets[:, d] += amplitude * np.sin(frequency * 2*np.pi * time_normalized.flatten() + phase)
        
        return targets
        
    def _laplacian(self, z):
        """Helper function to compute discrete Laplacian with periodic boundaries."""
        zU = np.roll(z, 1, axis=0)
        zD = np.roll(z, -1, axis=0)
        zL = np.roll(z, 1, axis=1)
        zR = np.roll(z, -1, axis=1)
        return (zU + zD + zL + zR - 4 * z)
        
    def visualize_data(self, positions, time_points, cell_features, targets):
        """
        Visualize generated data for inspection.
        
        Args:
            positions: Tensor of cell positions [batch_size, num_cells, 2]
            time_points: Tensor of time points [batch_size, num_cells, 1]
            cell_features: Tensor of cell features [batch_size, num_cells, feature_dim]
            targets: Target features [batch_size, num_cells, feature_dim]
        """
        # Convert to numpy
        positions = positions[0].numpy()
        time_points = time_points[0].numpy()
        features = cell_features[0].numpy()
        targets = targets[0].numpy()
        
        # Create figure
        plt.figure(figsize=(15, 12))
        
        # 1. Plot cell positions colored by time
        plt.subplot(2, 2, 1)
        scatter = plt.scatter(positions[:, 0], positions[:, 1], 
                             c=time_points, cmap='viridis', s=30, alpha=0.7)
        plt.colorbar(scatter, label='Time')
        plt.title('Cell Positions and Time')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        
        # 2. Plot a few feature dimensions
        plt.subplot(2, 2, 2)
        feature_dims = min(3, features.shape[1])
        for d in range(feature_dims):
            plt.scatter(positions[:, 0], positions[:, 1], 
                      c=features[:, d], cmap=f'plasma', s=30, alpha=0.7)
            plt.colorbar(label=f'Feature {d}')
            plt.title(f'Feature Dimension {d}')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            
            if d < feature_dims - 1:
                plt.figure(figsize=(8, 6))
        
        # 3. Plot temporal dynamics for a few cells
        plt.figure(figsize=(12, 6))
        sampled_cells = np.random.choice(positions.shape[0], min(5, positions.shape[0]), replace=False)
        
        # Sort by time
        sort_idx = np.argsort(time_points.flatten())
        times_sorted = time_points.flatten()[sort_idx]
        
        # Get features at each time for sampled cells
        for i, cell_idx in enumerate(sampled_cells):
            # Plot a few feature dimensions over time
            for d in range(min(3, features.shape[1])):
                plt.plot(time_points[cell_idx], features[cell_idx, d], 
                       'o-', label=f'Cell {i}, Feature {d}')
                
        plt.title('Temporal Dynamics of Features')
        plt.xlabel('Time')
        plt.ylabel('Feature Value')
        plt.legend()
        
        plt.tight_layout()
        plt.show()


# Training setup with morphogenesis data
def train_encoder_with_morphogenesis(encoder, 
                                    epochs=50, 
                                    batch_size=32, 
                                    num_cells=100, 
                                    feature_dim=64,
                                    log_interval=5):
    """Train encoder with morphogenesis data."""
    # Create data generator
    data_gen = MorphogenesisDataGenerator(
        spatial_size=100,
        max_time=100,
        feature_dim=feature_dim
    )
    
    # Create test model with our encoder
    test_model = EncoderTestModel(encoder, feature_dim)
    optimizer = torch.optim.Adam(test_model.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=0.00001
    )    
    criterion = nn.MSELoss()
    
    # For visualization and monitoring
    loss_history = []
    
    for epoch in range(epochs):
        # Generate a batch using morphogenesis data
        positions, time_points, cell_features, targets = data_gen.generate_batch(
            batch_size=batch_size, 
            num_cells=num_cells,
            method='mixed'
        )
        
        # Forward pass
        optimizer.zero_grad()
        outputs = test_model(positions, time_points, cell_features)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        loss_history.append(loss.item())
        
        if (epoch + 1) % log_interval == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    # Visualize a sample of the data
    sample_positions, sample_time_points, sample_features, sample_targets = data_gen.generate_batch(
        batch_size=1, 
        num_cells=num_cells,
        method='reaction_diffusion'
    )
    
    data_gen.visualize_data(
        sample_positions, 
        sample_time_points, 
        sample_features,
        sample_targets
    )
    
    return encoder


# Test model class
class EncoderTestModel(nn.Module):
    def __init__(self, encoding_module, feature_dim=64):
        super().__init__()
        self.encoder = encoding_module
        self.output_layer = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, positions, time_points, cell_features):
        encoded_features = self.encoder(positions, time_points, cell_features)
        return self.output_layer(encoded_features)

if __name__ == "__main__":
    # Create the encoder
    embedding_dim = 64
    encoder = NeRFSpatiotemporalEncoding(embedding_dim)
    
    # Train the encoder
    trained_encoder = train_encoder_with_morphogenesis(encoder, 200)
    
    # Save the trained encoder
    torch.save(trained_encoder.state_dict(), 'trained_encoder.pth')
    print("Encoder trained and saved successfully!")
