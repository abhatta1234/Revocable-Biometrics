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

class PairedEmbeddingsDataset(Dataset):
    """
    Dataset that loads paired embeddings from a text file.
    Each entry in the text file contains two paths to .npy files.
    """
    def __init__(self, pairs_file):
        """
        Args:
            pairs_file (str): Path to the text file containing paired .npy paths.
        """
        super().__init__()
        self.pairs = self._load_pairs(pairs_file)

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

class EmbeddingTransformer(nn.Module):
    """
    Learns a transformation from 512-d -> 512-d.
    You can tweak hidden dims, activation, etc.
    """

    def __init__(self, input_dim=512, hidden_dim=256, output_dim=512):
        super(EmbeddingTransformer, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------
# 3) Instantiate your dataset, model, and loss
# ------------------------------------------------------------------
# Load full dataset
dataset = PairedEmbeddingsDataset("/afs/crc.nd.edu/user/a/abhatta/Revocable_Journal_Extension/txt_pairs/model1_2.txt")
# Define split sizes (e.g., 80% train, 20% test)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size  # Ensure total matches

# Randomly split dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Print sizes
print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

#
model = EmbeddingTransformer()
model = model.cuda()  # if you have a GPU
optimizer = optim.Adam(model.parameters(), lr=1e-3)

#
# # Option 2: Cosine similarity loss (you want to maximize similarity,
# # so your objective is: L = 1 - cos(emb_pred, emb_tgt).
criterion_cos = nn.CosineEmbeddingLoss(margin=0.0)

# # If using CosineEmbeddingLoss, you need labels of +1 for "same"
# # and -1 for "different". For identical pairs: label = +1.
# # Alternatively, just do:
# #   L = 1 - F.cosine_similarity(emb_pred, emb_tgt, dim=1).mean()
# # in your training loop.
#
# # ------------------------------------------------------------------
# # 4) Training Loop
# # ------------------------------------------------------------------
num_epochs = 10  # or more

print("training the model....")
for epoch in tqdm(range(num_epochs)):
    model.train()
    total_loss = 0.0
    for emb_src, emb_tgt in tqdm(train_loader):
        emb_src = emb_src.cuda()
        emb_tgt = emb_tgt.cuda()

        optimizer.zero_grad()

        # Forward pass: transform the source to predicted target
        emb_pred = model(emb_src)

        # (b) If you want to use a simple "1 - cos" objective:
        # directly optimizing the cosine_similarity
        cos_sim = torch.nn.functional.cosine_similarity(emb_pred, emb_tgt, dim=1)
        loss = (1 - cos_sim).mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}]  Loss: {avg_loss:.4f}")


# ------------------------------------------------------------------
# 4) Eval
# ------------------------------------------------------------------

print("testing the model....")
model.eval()  # Set model to evaluation mode

sim_before = []
sim_after = []

with torch.no_grad():
    for emb_src, emb_tgt in tqdm(test_loader):
        emb_src = emb_src.cuda()
        emb_tgt = emb_tgt.cuda()

        # 1. Compute cosine similarity BEFORE transformation
        cos_before = F.cosine_similarity(emb_src, emb_tgt, dim=1)
        sim_before.extend(cos_before.cpu().numpy())

        # 2. Transform the source embedding
        emb_pred = model(emb_src)

        # 3. Normalize embeddings (if teacher embeddings are normalized)
        emb_pred = F.normalize(emb_pred, p=2, dim=1)

        # 4. Compute cosine similarity AFTER transformation
        cos_after = F.cosine_similarity(emb_pred, emb_tgt, dim=1)
        sim_after.extend(cos_after.cpu().numpy())

# ------------------------------------------------------------------
# Compute Average Similarity
# ------------------------------------------------------------------
avg_before = sum(sim_before) / len(sim_before)
avg_after = sum(sim_after) / len(sim_after)

print(f"\n[Evaluation] Average Cosine Similarity BEFORE transformation: {avg_before:.4f}")
print(f"[Evaluation] Average Cosine Similarity AFTER transformation: {avg_after:.4f}")
print(f"[Evaluation] Improvement: {avg_after - avg_before:.4f}")

# ------------------------------------------------------------------
# Plot Similarity Distribution
# ------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.hist(sim_before, bins=50, alpha=0.5, label="Before Transformation", color="red")
plt.hist(sim_after, bins=50, alpha=0.5, label="After Transformation", color="green")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.title("Cosine Similarity Distribution: Before vs After Transformation")
plt.legend()
plt.savefig("check.png")