import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as colors
from tqdm import tqdm
import random

class PairedEmbeddingsDataset(Dataset):
    """
    Dataset that loads paired embeddings from a text file.
    Each entry in the text file contains two paths to .npy files.
    """
    def __init__(self, pairs_file,fraction=1.0, random_seed=42):
        """
        Args:
            pairs_file (str): Path to the text file containing paired .npy paths.
        """
        super().__init__()
        self.pairs = self._load_pairs(pairs_file)

        if fraction < 1.0:
            random.seed(random_seed)
            sample_size = int(len(self.pairs) * fraction)
            self.pairs = random.sample(self.pairs, sample_size)
        print("📊📊📊 the total num of pairs used 📊📊📊 ", len(self.pairs))

    def _load_pairs(self, pairs_file):
        """Loads the list of (file1, file2) paths from a text file."""
        pairs = []
        with open(pairs_file, 'r') as f:
            for line in f:
                p1, p2 = line.strip().split(" ")
                pairs.append((p1, p2))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """Loads embeddings from file and returns as torch tensors."""
        path1, path2 = self.pairs[idx]

        # Load embeddings
        emb1 = np.load(path1).flatten()  # Shape: [512]
        emb2 = np.load(path2).flatten()  # Shape: [512]

        #print(emb1.shape,emb2.shape)

        # Convert to PyTorch tensors
        emb1 = torch.from_numpy(emb1).float()
        emb2 = torch.from_numpy(emb2).float()

        return emb1, emb2


# ------------------------------------------------------------------
# 2) The MLP Model
# ------------------------------------------------------------------


class DynamicEmbeddingTransformer(nn.Module):
    """
    Learns a transformation from input_dim to output_dim.
    Allows dynamic configuration of the number of hidden layers.
    """

    def __init__(self, input_dim=512, hidden_dim=256, output_dim=512, num_hidden_layers=2):
        """
        Args:
            input_dim (int): Dimensionality of the input.
            hidden_dim (int): Dimensionality of the hidden layers.
            output_dim (int): Dimensionality of the output.
            num_hidden_layers (int): Number of hidden layers.
                                      Must be at least 1.
        """
        super(DynamicEmbeddingTransformer, self).__init__()

        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be at least 1")

        layers = []
        # First layer: input to hidden
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))

        # Additional hidden layers (if any)
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        # Output layer: hidden to output
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------
# 3) Instantiate your dataset, model, and loss -- Frontier Learning
# ------------------------------------------------------------------

# Load full dataset
#full_dataset = PairedEmbeddingsDataset("/afs/crc.nd.edu/user/a/abhatta/Revocable_Journal_Extension/txt_pairs/model1_2.txt")
full_dataset = PairedEmbeddingsDataset("/afs/crc.nd.edu/user/a/abhatta/Revocable_Journal_Extension/txt_pairs/model1_2.txt", fraction=0.03)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Base model parameters
base_input_dim = 512
base_hidden_dim = 256
base_output_dim = 512

# Experiment configuration:
num_epochs = 10  # number of epochs per config
batch_size = 32

# Create scaling arrays:
num_linear_layers = np.array(range(1, 4)) # 1 to 4 Hidden dimension
model_scales = np.linspace(0.1, 1, num=5)       # 10% to 200%
dataset_scales = np.linspace(0.01, 1.0, num=5)     # 1% to 100%

# Container for recording results.
results = []
print("starte the frontier training loop......")
for num_layers in tqdm(num_linear_layers):
    for m_scale in tqdm(model_scales):
        # Compute scaled model parameters
        scaled_input_dim = int(base_input_dim)
        scaled_hidden_dim = int(base_hidden_dim * m_scale)
        scaled_output_dim = int(base_output_dim)

        for d_scale in tqdm(dataset_scales):
            # Create a subset of the full dataset based on the dataset fraction
            subset_size = int(len(full_dataset) * d_scale)
            if subset_size < 2:  # Skip if subset too small to split
                continue

            # Randomly split the selected subset into 80% train and 20% test
            subset, _ = random_split(full_dataset, [subset_size, len(full_dataset) - subset_size])
            train_size = int(0.8 * len(subset))
            test_size = len(subset) - train_size
            train_dataset, test_dataset = random_split(subset, [train_size, test_size])

            # Create DataLoaders for training and testing
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

            # Instantiate model with scaled parameters and move to device
            model = DynamicEmbeddingTransformer(input_dim=scaled_input_dim,
                                         hidden_dim=scaled_hidden_dim,
                                         output_dim=scaled_output_dim,
                                         num_hidden_layers=num_layers)


            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)

            # ---------------------------
            # Training Loop for this config
            # ---------------------------
            for epoch in range(num_epochs):
                model.train()
                total_loss = 0.0
                for emb_src, emb_tgt in train_loader:
                    emb_src = emb_src.to(device, non_blocking=True)
                    emb_tgt = emb_tgt.to(device, non_blocking=True)

                    optimizer.zero_grad()
                    emb_pred = model(emb_src)
                    loss = (1 - F.cosine_similarity(emb_pred, emb_tgt, dim=1)).mean()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                avg_loss = total_loss / len(train_loader)
                print(
                    f"Num_linear_layers:{num_layers},Model scale {m_scale:.2f}, Data scale {d_scale:.2f}, Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

            # ---------------------------
            # Evaluation Loop for this config
            # ---------------------------
            model.eval()
            sim_before = []
            sim_after = []
            with torch.no_grad():
                for emb_src, emb_tgt in test_loader:
                    emb_src = emb_src.to(device)
                    emb_tgt = emb_tgt.to(device)

                    # Cosine similarity before transformation
                    cos_before = F.cosine_similarity(emb_src, emb_tgt, dim=1)
                    sim_before.extend(cos_before.cpu().numpy())

                    # Transform the source embeddings
                    emb_pred = model(emb_src)
                    # Optionally, normalize the predictions if needed
                    emb_pred = F.normalize(emb_pred, p=2, dim=1)
                    cos_after = F.cosine_similarity(emb_pred, emb_tgt, dim=1)
                    sim_after.extend(cos_after.cpu().numpy())

            avg_before = np.mean(sim_before)
            avg_after = np.mean(sim_after)
            improvement = avg_after - avg_before

            print({
                "num_linear_layers":num_layers,
                "model_scale": m_scale,
                "dataset_scale": d_scale,
                "avg_before": avg_before,
                "avg_after": avg_after,
                "improvement": improvement
            })

            # Record the results for this configuration
            results.append({
                "num_linear_layers":num_layers,
                "model_scale": m_scale,
                "dataset_scale": d_scale,
                "avg_before": avg_before,
                "avg_after": avg_after,
                "improvement": improvement
            })

            # Clean up to free GPU memory
            del model
            torch.cuda.empty_cache()


def plot_heatmap_improvement_discrete(data, model_thresh=0.32, dataset_thresh=0.32, num_bins=6):
    """
    Create a discretized heatmap for each hidden-layer configuration where:
      - x-axis: model_scale,
      - y-axis: dataset_scale,
      - color: binned average similarity improvement.

    Additionally, a vertical line at the model_thresh and a horizontal line
    at the dataset_thresh are added to each subplot.
    """
    df = pd.DataFrame(data)
    layers = sorted(df['num_linear_layers'].unique())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, layer in enumerate(layers):
        ax = axes[i]
        subset = df[df['num_linear_layers'] == layer]
        pivot = subset.pivot_table(index='dataset_scale', columns='model_scale',
                                   values='improvement', aggfunc='mean')

        # Define extent for the heatmap axes
        extent = [pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()]

        # Define discrete bins for the improvement values
        imp_min, imp_max = pivot.values.min(), pivot.values.max()
        bins = np.linspace(imp_min, imp_max, num_bins + 1)
        norm = colors.BoundaryNorm(bins, ncolors=256)

        # Use a calmer colormap: cividis
        im = ax.imshow(pivot.values, origin='lower', aspect='auto',
                       extent=extent, cmap='PuBuGn', norm=norm)

        # Add threshold lines for model and dataset scales
        ax.axvline(x=model_thresh, color='red', linestyle='--', linewidth=2)
        ax.axhline(y=dataset_thresh, color='red', linestyle='--', linewidth=2)

        ax.set_title(f'{layer} Hidden Layer{"s" if layer > 1 else ""}; $n$ = 256', fontsize=14)
        ax.set_xlabel('Model Scale', fontsize=12)
        ax.set_ylabel('Dataset Scale', fontsize=12)
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig("accuracy_improvement_heatmap.png")


def get_threshold_point(pivot, metric_threshold):
    """
    Given a pivot table (with dataset_scale as index and model_scale as columns)
    and a metric threshold, return a tuple:
      (threshold_x, threshold_y, metric_value_at_point, status)

    status is:
      - 'low' if the threshold is lower than the smallest metric value (point set to bottom left)
      - 'high' if the threshold is higher than the maximum metric value (point set to top right)
      - 'normal' otherwise (the first point where the metric meets/exceeds the threshold)
    """
    x_min = pivot.columns.min()
    x_max = pivot.columns.max()
    y_min = pivot.index.min()
    y_max = pivot.index.max()
    min_val = pivot.values.min()
    max_val = pivot.values.max()

    if metric_threshold <= min_val:
        # Threshold is below the smallest metric value: use bottom left.
        return x_min, y_min, min_val, 'low'
    elif metric_threshold > max_val:
        # Threshold is not achieved anywhere: use top right.
        return x_max, y_max, max_val, 'high'
    else:
        # Search in ascending order (by dataset_scale then model_scale)
        for y in np.sort(pivot.index.values):
            for x in np.sort(pivot.columns.values):
                if pivot.loc[y, x] >= metric_threshold:
                    return x, y, pivot.loc[y, x], 'normal'
        # Fallback in case nothing is found (should not occur)
        return x_max, y_max, max_val, 'high'


def plot_combined_contours_with_threshold(data, metric_threshold, num_bins=6):
    """
    For each hidden-layer configuration, create a row with two 2D contour plots:
      - Column 1: Contour plot of 'avg_before' with a threshold marker.
      - Column 2: Contour plot of 'avg_after' with a threshold marker.

    For each plot, the function finds the threshold point as follows:
      - If the threshold is lower than all metric values, the marker is placed at the bottom left.
      - If the threshold is higher than all metric values, the marker is placed at the top right.
      - Otherwise, the first (smallest) (model_scale, dataset_scale) that meets/exceeds the threshold is used.

    A large red dot (s=150) marks the threshold point, and dashed lines are drawn:
      - For 'normal' or 'low': from the point to the left (min model scale) and bottom (min dataset scale).
      - For 'high': from the point to the right (max model scale) and top (max dataset scale).
    """
    df = pd.DataFrame(data)
    layers = sorted(df['num_linear_layers'].unique())
    num_layers = len(layers)

    # Create a figure with 2 columns per layer.
    fig = plt.figure(figsize=(12, 4 * num_layers))
    gs = gridspec.GridSpec(num_layers, 2, figure=fig)

    for i, layer in enumerate(layers):
        subset = df[df['num_linear_layers'] == layer]

        # Create pivot tables for the two metrics.
        pivot_before = subset.pivot_table(index='dataset_scale', columns='model_scale',
                                          values='avg_before', aggfunc='mean')
        pivot_after = subset.pivot_table(index='dataset_scale', columns='model_scale',
                                         values='avg_after', aggfunc='mean')

        # --- Column 1: 'Before' Contour Plot ---
        ax_before = fig.add_subplot(gs[i, 0])
        cont_before = ax_before.contourf(pivot_before.columns.values, pivot_before.index.values,
                                         pivot_before.values, levels=num_bins, cmap='PuBuGn')
        fig.colorbar(cont_before, ax=ax_before)
        ax_before.set_title(f'{layer} Hidden Layer{"s" if layer > 1 else ""} - Before')
        ax_before.set_xlabel('Model Scale')
        ax_before.set_ylabel('Dataset Scale')

        # Determine plot boundaries.
        x_min = pivot_before.columns.min()
        x_max = pivot_before.columns.max()
        y_min = pivot_before.index.min()
        y_max = pivot_before.index.max()

        # Compute threshold point for 'Before'
        x_thresh, y_thresh, val_thresh, status = get_threshold_point(pivot_before, metric_threshold)
        ax_before.scatter(x_thresh, y_thresh, color='red', s=150,
                          label=f'Threshold ({metric_threshold:.2f})')
        if status in ['normal', 'low']:
            # Project dashed lines to the left (x_min) and bottom (y_min)
            ax_before.plot([x_thresh, x_min], [y_thresh, y_thresh], 'r--', linewidth=2)
            ax_before.plot([x_thresh, x_thresh], [y_thresh, y_min], 'r--', linewidth=2)
        elif status == 'high':
            # Project dashed lines to the right (x_max) and top (y_max)
            ax_before.plot([x_thresh, x_max], [y_thresh, y_thresh], 'r--', linewidth=2)
            ax_before.plot([x_thresh, x_thresh], [y_thresh, y_max], 'r--', linewidth=2)
        ax_before.legend()

        # --- Column 2: 'After' Contour Plot ---
        ax_after = fig.add_subplot(gs[i, 1])
        cont_after = ax_after.contourf(pivot_after.columns.values, pivot_after.index.values,
                                       pivot_after.values, levels=num_bins, cmap='PuBuGn')
        fig.colorbar(cont_after, ax=ax_after)
        ax_after.set_title(f'{layer} Hidden Layer{"s" if layer > 1 else ""} - After')
        ax_after.set_xlabel('Model Scale')
        ax_after.set_ylabel('Dataset Scale')

        # Determine plot boundaries for 'After'
        x_min_a = pivot_after.columns.min()
        x_max_a = pivot_after.columns.max()
        y_min_a = pivot_after.index.min()
        y_max_a = pivot_after.index.max()

        # Compute threshold point for 'After'
        x_thresh_a, y_thresh_a, val_thresh_a, status_a = get_threshold_point(pivot_after, metric_threshold)
        ax_after.scatter(x_thresh_a, y_thresh_a, color='red', s=150,
                         label=f'Threshold ({metric_threshold:.2f})')
        if status_a in ['normal', 'low']:
            ax_after.plot([x_thresh_a, x_min_a], [y_thresh_a, y_thresh_a], 'r--', linewidth=2)
            ax_after.plot([x_thresh_a, x_thresh_a], [y_thresh_a, y_min_a], 'r--', linewidth=2)
        elif status_a == 'high':
            ax_after.plot([x_thresh_a, x_max_a], [y_thresh_a, y_thresh_a], 'r--', linewidth=2)
            ax_after.plot([x_thresh_a, x_thresh_a], [y_thresh_a, y_max_a], 'r--', linewidth=2)
        ax_after.legend()

    plt.tight_layout()
    plt.savefig("accuracy_improvement_heatmap.png")


plot_combined_contours_with_threshold(results, metric_threshold=0.32,num_bins=6)

