import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from datetime import datetime
import colorama
from colorama import Fore, Back, Style

# Initialize colorama
colorama.init(autoreset=True)

# Symbols for section headers
SYMBOL_DATASET = "📊"
SYMBOL_MODEL = "🧠"
SYMBOL_TRAINING = "🔥"
SYMBOL_EVALUATION = "🔍"
SYMBOL_RESULTS = "🏆"
SYMBOL_CONFIG = "⚙️"
SYMBOL_TIME = "⏱️"
SYMBOL_ERROR = "❌"
SYMBOL_SUCCESS = "✅"
SYMBOL_WARNING = "⚠️"


def print_section_header(symbol, text):
    """Print a formatted section header with symbol"""
    print("\n" + "=" * 80)
    print(f"{Fore.CYAN}{Style.BRIGHT}{symbol} {text} {symbol}")
    print("=" * 80)


def print_info(text):
    """Print info message"""
    print(f"{Fore.GREEN}[INFO] {text}")


def print_warning(text):
    """Print warning message"""
    print(f"{Fore.YELLOW}{SYMBOL_WARNING} [WARNING] {text}")


def print_error(text):
    """Print error message"""
    print(f"{Fore.RED}{SYMBOL_ERROR} [ERROR] {text}")


def print_success(text):
    """Print success message"""
    print(f"{Fore.GREEN}{SYMBOL_SUCCESS} {text}")


def print_time(start_time, operation):
    """Print time taken for an operation"""
    elapsed = time.time() - start_time
    print(f"{Fore.MAGENTA}{SYMBOL_TIME} {operation} completed in {elapsed:.2f} seconds")


def print_progress(current, total, prefix="", suffix="", decimals=1, length=50, fill="█"):
    """Print a progress bar"""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (current / float(total)))
    filled_length = int(length * current // total)
    bar = fill * filled_length + "-" * (length - filled_length)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end="\r")
    if current == total:
        print()


# ------------------------------------------------------------------
# 1) In-Memory Dataset
# ------------------------------------------------------------------
print_section_header(SYMBOL_DATASET, "DATASET INITIALIZATION")


class PairedEmbeddingsDatasetInMemory(Dataset):
    """
    Loads all paired embeddings directly into memory at initialization.
    Each line of `pairs_file` has two .npy file paths: path1 path2
    """

    def __init__(self, pairs_file):
        super().__init__()
        self.emb_src = []
        self.emb_tgt = []

        # Read file paths
        print_info(f"Reading file paths from: {pairs_file}")
        start_time = time.time()
        try:
            with open(pairs_file, 'r') as f:
                lines = [line.strip().split() for line in f]
            print_success(f"Found {len(lines)} embedding pairs")
        except FileNotFoundError:
            print_error(f"File not found: {pairs_file}")
            raise

        # Load each .npy into memory
        print_info("Loading embeddings into memory...")
        load_start = time.time()
        for i, (p1, p2) in enumerate(tqdm(lines, desc=f"{Fore.BLUE}Loading data")):
            try:
                arr1 = np.load(p1).astype(np.float32).flatten()
                arr2 = np.load(p2).astype(np.float32).flatten()
                self.emb_src.append(arr1)
                self.emb_tgt.append(arr2)
            except Exception as e:
                print_error(f"Error loading files {p1} or {p2}: {str(e)}")
                continue

        # Stack into big numpy arrays, then convert to Tensors
        print_info("Converting to PyTorch tensors...")
        self.emb_src = torch.from_numpy(np.stack(self.emb_src))  # shape: [N, 512]
        self.emb_tgt = torch.from_numpy(np.stack(self.emb_tgt))  # shape: [N, 512]
        print_time(load_start, "Data loading")

        # Print shape information
        print_info(f"Source embeddings shape: {self.emb_src.shape}")
        print_info(f"Target embeddings shape: {self.emb_tgt.shape}")
        print_time(start_time, "Dataset initialization")

    def __len__(self):
        return self.emb_src.size(0)

    def __getitem__(self, idx):
        # Return preloaded Tensors
        return self.emb_src[idx], self.emb_tgt[idx]


# ------------------------------------------------------------------
# 2) The MLP Model
# ------------------------------------------------------------------
print_section_header(SYMBOL_MODEL, "MODEL ARCHITECTURE")


class EmbeddingTransformer(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

        # Print model configuration
        print_info(f"EmbeddingTransformer initialized with:")
        print_info(f"  - Input dimension: {input_dim}")
        print_info(f"  - Hidden dimension: {hidden_dim}")
        print_info(f"  - Output dimension: {output_dim}")

        # Calculate number of parameters
        param_count = sum(p.numel() for p in self.parameters())
        print_info(f"  - Total parameters: {param_count:,}")

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------
# 3) Configuration and Setup
# ------------------------------------------------------------------
print_section_header(SYMBOL_CONFIG, "CONFIGURATION")

# Dataset configuration
pairs_file = "/afs/crc.nd.edu/user/a/abhatta/Revocable_Journal_Extension/txt_pairs/model2_4.txt"
print_info(f"Pairs file: {pairs_file}")

# Training configuration
batch_size = 64
num_workers = 4
learning_rate = 1e-3
num_epochs = 10
train_split = 0.8

print_info(f"Batch size: {batch_size}")
print_info(f"Number of workers: {num_workers}")
print_info(f"Learning rate: {learning_rate}")
print_info(f"Number of epochs: {num_epochs}")
print_info(f"Train/test split: {train_split}/{1 - train_split}")

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print_info(f"Using device: {device}")
if device.type == "cuda":
    print_info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print_info(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

# ------------------------------------------------------------------
# 4) Instantiate dataset and prepare loaders
# ------------------------------------------------------------------
print_section_header(SYMBOL_DATASET, "DATASET PREPARATION")

start_time = time.time()
print_info("Creating dataset...")
dataset = PairedEmbeddingsDatasetInMemory(pairs_file)

# Split dataset
print_info("Splitting dataset into train and test sets...")
train_size = int(train_split * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
print_success(f"Train set size: {len(train_dataset)}")
print_success(f"Test set size: {len(test_dataset)}")

# Create DataLoaders
print_info("Creating DataLoaders...")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)
print_success(f"Train batches: {len(train_loader)}")
print_success(f"Test batches: {len(test_loader)}")
print_time(start_time, "Data preparation")

# ------------------------------------------------------------------
# 5) Initialize model and optimizer
# ------------------------------------------------------------------
print_section_header(SYMBOL_MODEL, "MODEL INITIALIZATION")

start_time = time.time()
print_info("Initializing model...")
model = EmbeddingTransformer().to(device)
print_info("Initializing optimizer...")
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print_time(start_time, "Model setup")

# ------------------------------------------------------------------
# 6) Training loop
# ------------------------------------------------------------------
print_section_header(SYMBOL_TRAINING, "TRAINING")
print_info(f"Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print_info(f"Training for {num_epochs} epochs")

# Track metrics
train_losses = []
epoch_times = []

for epoch in range(num_epochs):
    epoch_start = time.time()
    model.train()
    total_loss = 0.0

    print_info(f"Epoch {epoch + 1}/{num_epochs}")

    # Initialize progress bar
    batch_bar = tqdm(total=len(train_loader),
                     desc=f"{Fore.BLUE}Training",
                     bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET))

    for i, (emb_src, emb_tgt) in enumerate(train_loader):
        emb_src = emb_src.to(device, non_blocking=True)
        emb_tgt = emb_tgt.to(device, non_blocking=True)

        optimizer.zero_grad()

        emb_pred = model(emb_src)
        # Loss = 1 - cosine_similarity
        loss = (1 - F.cosine_similarity(emb_pred, emb_tgt, dim=1)).mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Update progress bar with current loss
        batch_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        batch_bar.update()

    batch_bar.close()

    # Calculate average loss for the epoch
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    # Calculate epoch time
    epoch_time = time.time() - epoch_start
    epoch_times.append(epoch_time)

    # Print epoch summary
    print_success(f"Epoch [{epoch + 1}/{num_epochs}] - " +
                  f"Loss: {Fore.YELLOW}{avg_loss:.4f}{Fore.RESET} - " +
                  f"Time: {epoch_time:.2f}s - " +
                  f"ETA: {epoch_time * (num_epochs - epoch - 1):.2f}s")

    # Show GPU memory usage if available
    if device.type == "cuda":
        print_info(f"GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB / " +
                   f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Print training summary
total_train_time = sum(epoch_times)
print_success(f"Total training time: {total_train_time:.2f} seconds")
print_success(f"Average epoch time: {np.mean(epoch_times):.2f} seconds")
print_success(f"Final training loss: {train_losses[-1]:.4f}")

# ------------------------------------------------------------------
# 7) Evaluate the model
# ------------------------------------------------------------------
print_section_header(SYMBOL_EVALUATION, "EVALUATION")

eval_start = time.time()
model.eval()
sim_before = []
sim_after = []

print_info("Evaluating model on test set...")

# Initialize progress bar for evaluation
eval_bar = tqdm(total=len(test_loader),
                desc=f"{Fore.BLUE}Evaluating",
                bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET))

with torch.no_grad():
    for emb_src, emb_tgt in test_loader:
        emb_src = emb_src.to(device, non_blocking=True)
        emb_tgt = emb_tgt.to(device, non_blocking=True)

        # Before transformation
        cos_before = F.cosine_similarity(emb_src, emb_tgt, dim=1)
        sim_before.extend(cos_before.cpu().numpy())

        # After transformation
        emb_pred = model(emb_src)
        emb_pred = F.normalize(emb_pred, p=2, dim=1)
        cos_after = F.cosine_similarity(emb_pred, emb_tgt, dim=1)
        sim_after.extend(cos_after.cpu().numpy())

        # Update progress bar
        eval_bar.update()

eval_bar.close()

# Calculate metrics
avg_before = np.mean(sim_before)
avg_after = np.mean(sim_after)
improvement = avg_after - avg_before
std_before = np.std(sim_before)
std_after = np.std(sim_after)

print_time(eval_start, "Evaluation")

# ------------------------------------------------------------------
# 8) Results and Visualization
# ------------------------------------------------------------------
print_section_header(SYMBOL_RESULTS, "RESULTS SUMMARY")

# Print results
print_info(f"Test samples: {len(sim_before)}")
print_info(f"Avg Cosine Similarity BEFORE: {Fore.YELLOW}{avg_before:.4f}{Fore.RESET} (std: {std_before:.4f})")
print_info(f"Avg Cosine Similarity AFTER:  {Fore.YELLOW}{avg_after:.4f}{Fore.RESET} (std: {std_after:.4f})")

# Print improvement with color based on value
if improvement > 0.1:
    print_success(f"Improvement: {Fore.GREEN}{improvement:.4f}{Fore.RESET} (+{improvement * 100:.2f}%)")
elif improvement > 0:
    print_info(f"Improvement: {Fore.BLUE}{improvement:.4f}{Fore.RESET} (+{improvement * 100:.2f}%)")
else:
    print_warning(f"No improvement: {Fore.RED}{improvement:.4f}{Fore.RESET}")

# Additional statistics
print_info(f"Min similarity BEFORE: {np.min(sim_before):.4f}")
print_info(f"Min similarity AFTER: {np.min(sim_after):.4f}")


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
plt.savefig("check_1.png")




# ------------------------------------------------------------------
# 7) Evaluate the on the unknown model mapping
# ------------------------------------------------------------------
print_section_header(SYMBOL_EVALUATION, "EVALUATION for the UNKNOWN Mappings")


# Dataset configuration
unknown_map_pairs_file = "/afs/crc.nd.edu/user/a/abhatta/Revocable_Journal_Extension/txt_pairs/model2_5.txt"
print_info(f"Pairs file: {unknown_map_pairs_file}")
unknown_map_dataset = PairedEmbeddingsDatasetInMemory(unknown_map_pairs_file)

unknown_map_loader = DataLoader(unknown_map_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)

eval_start = time.time()
model.eval()
sim_before = []
sim_after = []

print_info("Evaluating model on test set...")

# Initialize progress bar for evaluation
eval_bar = tqdm(total=len(unknown_map_loader),
                desc=f"{Fore.BLUE}Evaluating",
                bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET))

with torch.no_grad():
    for emb_src, emb_tgt in unknown_map_loader:
        emb_src = emb_src.to(device, non_blocking=True)
        emb_tgt = emb_tgt.to(device, non_blocking=True)

        # Before transformation
        cos_before = F.cosine_similarity(emb_src, emb_tgt, dim=1)
        sim_before.extend(cos_before.cpu().numpy())

        # After transformation
        emb_pred = model(emb_src)
        emb_pred = F.normalize(emb_pred, p=2, dim=1)
        cos_after = F.cosine_similarity(emb_pred, emb_tgt, dim=1)
        sim_after.extend(cos_after.cpu().numpy())

        # Update progress bar
        eval_bar.update()

eval_bar.close()

# Calculate metrics
avg_before = np.mean(sim_before)
avg_after = np.mean(sim_after)
improvement = avg_after - avg_before
std_before = np.std(sim_before)
std_after = np.std(sim_after)

print_time(eval_start, "Evaluation for the unknowns")

# ------------------------------------------------------------------
# 8) Results and Visualization
# ------------------------------------------------------------------
print_section_header(SYMBOL_RESULTS, "RESULTS SUMMARY")

# Print results
print_info(f"Test samples: {len(sim_before)}")
print_info(f"Avg Cosine Similarity BEFORE: {Fore.YELLOW}{avg_before:.4f}{Fore.RESET} (std: {std_before:.4f})")
print_info(f"Avg Cosine Similarity AFTER:  {Fore.YELLOW}{avg_after:.4f}{Fore.RESET} (std: {std_after:.4f})")

# Print improvement with color based on value
if improvement > 0.1:
    print_success(f"Improvement: {Fore.GREEN}{improvement:.4f}{Fore.RESET} (+{improvement * 100:.2f}%)")
elif improvement > 0:
    print_info(f"Improvement: {Fore.BLUE}{improvement:.4f}{Fore.RESET} (+{improvement * 100:.2f}%)")
else:
    print_warning(f"No improvement: {Fore.RED}{improvement:.4f}{Fore.RESET}")

# Additional statistics
print_info(f"Min similarity BEFORE: {np.min(sim_before):.4f}")
print_info(f"Min similarity AFTER: {np.min(sim_after):.4f}")


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
plt.savefig("check_unknown.png")