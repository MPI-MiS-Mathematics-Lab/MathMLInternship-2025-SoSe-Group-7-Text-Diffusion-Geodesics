#%%
"""
Language Model Curriculum Training based on Diffusion Geodesics

This module implements curriculum learning for language model fine-tuning using
the diffusion geometry structure of a text corpus. The key idea is to leverage
the Markov chain representation to sample training paths that start from low-entropy
(general/foundational) documents and progressively move to high-entropy 
(specialized/complex) documents.

Key Features:
- Curriculum-based batch sampling using Markov transition probabilities
- Progressive difficulty adjustment based on training progress
- Path-based sampling that respects semantic structure
- Baseline random sampling for comparison
- Configurable path depth and batch size parameters
- Training metrics and convergence tracking

The curriculum strategy should lead to faster convergence and better generalization
compared to random batch sampling.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from typing import List, Tuple, Dict
import json


def create_train_val_split(n_docs: int, val_ratio: float = 0.1, seed: int = 42) -> Tuple[List[int], List[int]]:
    """
    Create train/validation split indices.
    
    Args:
        n_docs: Total number of documents
        val_ratio: Fraction of documents to use for validation
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_indices, val_indices)
    """
    np.random.seed(seed)
    all_indices = np.arange(n_docs)
    np.random.shuffle(all_indices)
    
    n_val = int(n_docs * val_ratio)
    val_indices = all_indices[:n_val].tolist()
    train_indices = all_indices[n_val:].tolist()
    
    return train_indices, val_indices


#%%
class CurriculumSampler:
    """
    Samples training batches based on curriculum learning strategy using Markov chain.
    
    Uses centrality (stationary distribution probability) to order documents:
    - High centrality = central/common documents (easy, foundational)
    - Low centrality = peripheral/specialized documents (hard, complex)
    
    Instead of hard thresholds, samples from a Beta distribution over document
    difficulty ranks. The distribution shifts from favoring easy documents to
    favoring hard documents over epochs, while always keeping easy documents
    in the left tail with decreasing probability.
    """
    
    def __init__(self, 
                 markov_chain: np.ndarray,
                 stationary_distribution: np.ndarray,
                 n_epochs: int = 10,
                 batch_size: int = 32,
                 path_depth_min: int = 1,
                 path_depth_max: int = 5,
                 curriculum_schedule: str = "linear",
                 hard_emphasis: float = 0.7,
                 initial_concentration: float = 5.0,
                 final_concentration: float = 2.0):
        """
        Args:
            markov_chain: Transition probability matrix (n_docs x n_docs), after diffusion
            stationary_distribution: Stationary distribution (centrality) for each document
            n_epochs: Total number of training epochs
            batch_size: Target batch size
            path_depth_min: Minimum path length to sample
            path_depth_max: Maximum path length to sample
            curriculum_schedule: How to adjust distribution ("linear", "exponential", "cosine")
            hard_emphasis: Controls how quickly distribution shifts toward hard docs (0.5-0.9)
            initial_concentration: Beta distribution concentration at start (higher = more peaked on easy)
            final_concentration: Beta distribution concentration at end (lower = more spread out)
        """
        self.markov_chain = markov_chain
        self.stationary_distribution = stationary_distribution
        self.centrality = stationary_distribution  # Alias for clarity
        self.n_docs = len(stationary_distribution)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.path_depth_min = path_depth_min
        self.path_depth_max = path_depth_max
        self.curriculum_schedule = curriculum_schedule
        self.hard_emphasis = hard_emphasis
        self.initial_concentration = initial_concentration
        self.final_concentration = final_concentration
        
        # Sort documents by centrality (high to low = easy to hard)
        # Index 0 = easiest (highest centrality), Index n-1 = hardest (lowest centrality)
        self.centrality_order = np.argsort(stationary_distribution)[::-1]  # Descending
        
        # Create rank array: rank[doc_id] = position in difficulty order (0=easiest)
        self.difficulty_rank = np.zeros(self.n_docs, dtype=int)
        for rank, doc_id in enumerate(self.centrality_order):
            self.difficulty_rank[doc_id] = rank
        
        # Track sampling statistics
        self.sampled_docs = []
        self.sampled_centralities = []
        
    def get_sampling_distribution_params(self, epoch: int) -> Tuple[float, float]:
        """
        Get Beta distribution parameters (alpha, beta) for current epoch.
        
        The distribution is over normalized difficulty rank [0, 1]:
        - 0 = easiest document (highest centrality)
        - 1 = hardest document (lowest centrality)
        
        Early epochs: distribution peaked toward 0 (easy docs)
        Later epochs: distribution shifts toward 1 (hard docs) but maintains left tail
        
        Returns:
            Tuple of (alpha, beta) for Beta distribution
        """
        progress = epoch / self.n_epochs
        
        # Apply curriculum schedule to get adjusted progress
        if self.curriculum_schedule == "linear":
            adjusted_progress = progress
        elif self.curriculum_schedule == "exponential":
            adjusted_progress = (np.exp(progress * 2) - 1) / (np.exp(2) - 1)
        elif self.curriculum_schedule == "cosine":
            adjusted_progress = (1 - np.cos(progress * np.pi)) / 2
        else:
            adjusted_progress = progress
        
        # Interpolate concentration (decreases over time for more spread)
        concentration = self.initial_concentration - adjusted_progress * (self.initial_concentration - self.final_concentration)
        
        # Target mean of Beta distribution shifts from easy to hard
        # Start around 0.1-0.2 (mostly easy), end around 0.6-0.8 (mostly hard)
        # hard_emphasis controls how far the mean shifts
        target_mean_start = 0.15  # Early: mostly easy docs
        target_mean_end = 0.3 + 0.5 * self.hard_emphasis  # End: shifted toward hard
        
        target_mean = target_mean_start + adjusted_progress * (target_mean_end - target_mean_start)
        
        # Beta distribution: mean = alpha / (alpha + beta)
        # Given mean μ and concentration κ = alpha + beta:
        # alpha = μ * κ, beta = (1 - μ) * κ
        alpha = target_mean * concentration
        beta = (1 - target_mean) * concentration
        
        # Ensure valid parameters (> 0)
        alpha = max(0.5, alpha)
        beta = max(0.5, beta)
        
        return alpha, beta
    
    def get_centrality_threshold(self, epoch: int) -> float:
        """
        Get the median of the sampling distribution as an approximate threshold.
        Used for visualization/compatibility - actual sampling uses full distribution.
        """
        alpha, beta = self.get_sampling_distribution_params(epoch)
        # Approximate median of Beta distribution
        median_rank = (alpha - 1/3) / (alpha + beta - 2/3) if alpha > 1 and beta > 1 else alpha / (alpha + beta)
        median_rank = np.clip(median_rank, 0, 1)
        
        # Convert rank to centrality
        median_doc_idx = int(median_rank * (self.n_docs - 1))
        return self.centrality[self.centrality_order[median_doc_idx]]
    
    def sample_starting_document(self, epoch: int) -> int:
        """
        Sample a starting document from curriculum distribution.
        
        Uses Beta distribution over difficulty ranks, where:
        - Rank 0 = easiest (highest centrality)
        - Rank n-1 = hardest (lowest centrality)
        
        The distribution shifts toward harder documents over epochs,
        but easy documents always remain in the left tail.
        """
        alpha, beta = self.get_sampling_distribution_params(epoch)
        
        # Sample from Beta distribution (gives value in [0, 1])
        sampled_quantile = np.random.beta(alpha, beta)
        
        # Convert to document index in difficulty order
        doc_rank = int(sampled_quantile * (self.n_docs - 1))
        doc_rank = np.clip(doc_rank, 0, self.n_docs - 1)
        
        # Get actual document ID
        return self.centrality_order[doc_rank]
    
    def sample_path(self, start_doc: int, depth: int) -> List[int]:
        """
        Sample a path through the Markov chain starting from start_doc.
        Uses transition probabilities to select next documents.
        """
        path = [start_doc]
        current_doc = start_doc
        
        for _ in range(depth - 1):
            # Get transition probabilities from current document
            transition_probs = self.markov_chain[:, current_doc].copy()
            
            # Avoid self-loops by zeroing out current document
            transition_probs[current_doc] = 0
            
            # Normalize
            if transition_probs.sum() > 0:
                transition_probs = transition_probs / transition_probs.sum()
                # Sample next document
                next_doc = np.random.choice(self.n_docs, p=transition_probs)
                path.append(next_doc)
                current_doc = next_doc
            else:
                # No valid transitions, stop path
                break
                
        return path
    
    def sample_curriculum_batch(self, epoch: int) -> List[int]:
        """
        Sample a batch of documents for training using curriculum strategy.
        
        Samples starting documents from a Beta distribution over difficulty ranks
        that shifts from easy to hard over epochs. Then follows Markov chain paths.
        """
        batch_docs = []
        
        while len(batch_docs) < self.batch_size:
            # Sample path depth
            path_depth = np.random.randint(self.path_depth_min, self.path_depth_max + 1)
            
            # Sample starting document from curriculum distribution
            start_doc = self.sample_starting_document(epoch)
            
            # Sample path through diffused Markov chain
            path = self.sample_path(start_doc, path_depth)
            batch_docs.extend(path)
        
        # Trim to exact batch size
        batch_docs = batch_docs[:self.batch_size]
        
        # Track statistics
        self.sampled_docs.extend(batch_docs)
        self.sampled_centralities.extend([self.centrality[doc] for doc in batch_docs])
        
        return batch_docs
    
    def reset_statistics(self):
        """Reset sampling statistics."""
        self.sampled_docs = []
        self.sampled_centralities = []
    
    def visualize_sampling_distribution(self, epochs_to_show: List[int] = None):
        """
        Visualize how the sampling distribution changes over epochs.
        """
        import matplotlib.pyplot as plt
        from scipy import stats
        
        if epochs_to_show is None:
            epochs_to_show = [0, self.n_epochs // 4, self.n_epochs // 2, 
                             3 * self.n_epochs // 4, self.n_epochs - 1]
        
        x = np.linspace(0, 1, 200)
        
        plt.figure(figsize=(12, 5))
        
        # Plot 1: PDF over difficulty rank
        plt.subplot(1, 2, 1)
        for epoch in epochs_to_show:
            alpha, beta = self.get_sampling_distribution_params(epoch)
            y = stats.beta.pdf(x, alpha, beta)
            plt.plot(x, y, label=f'Epoch {epoch} (α={alpha:.2f}, β={beta:.2f})')
        
        plt.xlabel('Difficulty Rank (0=Easy, 1=Hard)')
        plt.ylabel('Probability Density')
        plt.title('Curriculum Sampling Distribution Over Epochs')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Expected difficulty over epochs
        plt.subplot(1, 2, 2)
        epochs = range(self.n_epochs)
        means = []
        stds = []
        for epoch in epochs:
            alpha, beta = self.get_sampling_distribution_params(epoch)
            mean = alpha / (alpha + beta)
            std = np.sqrt(alpha * beta / ((alpha + beta)**2 * (alpha + beta + 1)))
            means.append(mean)
            stds.append(std)
        
        means = np.array(means)
        stds = np.array(stds)
        
        plt.plot(epochs, means, 'b-', linewidth=2, label='Mean Difficulty')
        plt.fill_between(epochs, means - stds, means + stds, alpha=0.3, label='±1 Std Dev')
        plt.xlabel('Epoch')
        plt.ylabel('Expected Difficulty Rank')
        plt.title('Curriculum Progression')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('curriculum_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()


class RandomSampler:
    """
    Baseline random sampling strategy for comparison.
    """
    
    def __init__(self, n_docs: int, batch_size: int = 32):
        self.n_docs = n_docs
        self.batch_size = batch_size
        self.sampled_docs = []
        
    def sample_batch(self) -> List[int]:
        """Sample a random batch of documents."""
        batch = np.random.choice(self.n_docs, size=self.batch_size, replace=False)
        self.sampled_docs.extend(batch.tolist())
        return batch.tolist()
    
    def reset_statistics(self):
        """Reset sampling statistics."""
        self.sampled_docs = []


class TextDataset(Dataset):
    """
    Dataset for language model training with curriculum-based sampling.
    """
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }


class CurriculumTrainer:
    """
    Trainer that implements curriculum learning for language models.
    Uses centrality-based curriculum sampling.
    """
    
    def __init__(self,
                 model,
                 tokenizer,
                 df_corpus: pd.DataFrame,
                 sampler: CurriculumSampler,
                 val_indices: List[int] = None,
                 learning_rate: float = 5e-5,
                 n_epochs: int = 10,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.df_corpus = df_corpus
        self.sampler = sampler
        self.val_indices = val_indices if val_indices is not None else []
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.device = device
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Mixed precision training
        self.scaler = GradScaler()
        self.use_amp = device == 'cuda'
        
        # Track training metrics
        self.train_losses = []
        self.epoch_losses = []
        self.val_losses = []
        self.batch_centralities = []
        
    def train_epoch(self, epoch: int):
        """Train one epoch using curriculum sampling."""
        self.model.train()
        epoch_loss = 0
        n_batches = len(self.df_corpus) // self.sampler.batch_size
        
        for batch_idx in tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{self.n_epochs}"):
            # Sample batch using curriculum strategy
            doc_indices = self.sampler.sample_curriculum_batch(epoch)
            
            # Get texts for sampled documents
            batch_texts = [self.df_corpus.iloc[idx]['text'] for idx in doc_indices]
            
            # Tokenize
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                max_length=512,
                padding='max_length',
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            # MLM: randomly mask 15% of tokens
            labels = input_ids.clone()
            rand = torch.rand(input_ids.shape, device=self.device)
            mask_arr = (rand < 0.15) * (input_ids != self.tokenizer.pad_token_id)
            
            for i in range(input_ids.shape[0]):
                selection = torch.flatten(mask_arr[i].nonzero()).tolist()
                input_ids[i, selection] = self.tokenizer.mask_token_id
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
            
            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            self.train_losses.append(loss.item())
            
            # Track average centrality of batch
            batch_centrality = np.mean([self.sampler.centrality[idx] for idx in doc_indices])
            self.batch_centralities.append(batch_centrality)
        
        avg_epoch_loss = epoch_loss / n_batches
        self.epoch_losses.append(avg_epoch_loss)
        
        return avg_epoch_loss
    
    def evaluate(self, batch_size: int = 32) -> float:
        """Evaluate model on validation set."""
        if len(self.val_indices) == 0:
            return float('nan')
        
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(self.val_indices), batch_size):
                batch_indices = self.val_indices[i:i + batch_size]
                batch_texts = [self.df_corpus.iloc[idx]['text'] for idx in batch_indices]
                
                encodings = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    max_length=512,
                    padding='max_length',
                    return_tensors='pt'
                )
                
                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)
                
                # MLM: randomly mask 15% of tokens (fixed seed for consistency)
                labels = input_ids.clone()
                torch.manual_seed(42)  # Fixed seed for reproducible validation
                rand = torch.rand(input_ids.shape, device=self.device)
                mask_arr = (rand < 0.15) * (input_ids != self.tokenizer.pad_token_id)
                
                for j in range(input_ids.shape[0]):
                    selection = torch.flatten(mask_arr[j].nonzero()).tolist()
                    input_ids[j, selection] = self.tokenizer.mask_token_id
                
                with autocast(enabled=self.use_amp):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                
                total_loss += outputs.loss.item()
                n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else float('nan')
    
    def train(self):
        """Train the model for n_epochs."""
        print(f"Training on {self.device} (mixed precision: {self.use_amp})")
        print(f"Validation set size: {len(self.val_indices)} documents")
        
        for epoch in range(self.n_epochs):
            epoch_loss = self.train_epoch(epoch)
            val_loss = self.evaluate()
            self.val_losses.append(val_loss)
            print(f"Epoch {epoch+1}/{self.n_epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        return self.model


class RandomTrainer:
    """
    Baseline trainer with random sampling.
    """
    
    def __init__(self,
                 model,
                 tokenizer,
                 df_corpus: pd.DataFrame,
                 sampler: RandomSampler,
                 val_indices: List[int] = None,
                 learning_rate: float = 5e-5,
                 n_epochs: int = 10,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.df_corpus = df_corpus
        self.sampler = sampler
        self.val_indices = val_indices if val_indices is not None else []
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.device = device
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Mixed precision training
        self.scaler = GradScaler()
        self.use_amp = device == 'cuda'
        
        # Track training metrics
        self.train_losses = []
        self.epoch_losses = []
        self.val_losses = []
        
    def train_epoch(self, epoch: int):
        """Train one epoch using random sampling."""
        self.model.train()
        epoch_loss = 0
        n_batches = len(self.df_corpus) // self.sampler.batch_size
        
        for batch_idx in tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{self.n_epochs}"):
            # Sample random batch
            doc_indices = self.sampler.sample_batch()
            
            # Get texts for sampled documents
            batch_texts = [self.df_corpus.iloc[idx]['text'] for idx in doc_indices]
            
            # Tokenize
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                max_length=512,
                padding='max_length',
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            # MLM: randomly mask 15% of tokens
            labels = input_ids.clone()
            rand = torch.rand(input_ids.shape, device=self.device)
            mask_arr = (rand < 0.15) * (input_ids != self.tokenizer.pad_token_id)
            
            for i in range(input_ids.shape[0]):
                selection = torch.flatten(mask_arr[i].nonzero()).tolist()
                input_ids[i, selection] = self.tokenizer.mask_token_id
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
            
            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            self.train_losses.append(loss.item())
        
        avg_epoch_loss = epoch_loss / n_batches
        self.epoch_losses.append(avg_epoch_loss)
        
        return avg_epoch_loss
    
    def evaluate(self, batch_size: int = 32) -> float:
        """Evaluate model on validation set."""
        if len(self.val_indices) == 0:
            return float('nan')
        
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(self.val_indices), batch_size):
                batch_indices = self.val_indices[i:i + batch_size]
                batch_texts = [self.df_corpus.iloc[idx]['text'] for idx in batch_indices]
                
                encodings = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    max_length=512,
                    padding='max_length',
                    return_tensors='pt'
                )
                
                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)
                
                # MLM: randomly mask 15% of tokens (fixed seed for consistency)
                labels = input_ids.clone()
                torch.manual_seed(42)  # Fixed seed for reproducible validation
                rand = torch.rand(input_ids.shape, device=self.device)
                mask_arr = (rand < 0.15) * (input_ids != self.tokenizer.pad_token_id)
                
                for j in range(input_ids.shape[0]):
                    selection = torch.flatten(mask_arr[j].nonzero()).tolist()
                    input_ids[j, selection] = self.tokenizer.mask_token_id
                
                with autocast(enabled=self.use_amp):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                
                total_loss += outputs.loss.item()
                n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else float('nan')
    
    def train(self):
        """Train the model for n_epochs."""
        print(f"Training on {self.device} (mixed precision: {self.use_amp})")
        print(f"Validation set size: {len(self.val_indices)} documents")
        
        for epoch in range(self.n_epochs):
            epoch_loss = self.train_epoch(epoch)
            val_loss = self.evaluate()
            self.val_losses.append(val_loss)
            print(f"Epoch {epoch+1}/{self.n_epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        return self.model


#%%
def compute_perplexity_by_difficulty(model, tokenizer, df_corpus, centrality, 
                                      n_bins=3, n_samples_per_bin=100, device='cuda'):
    """
    Compute perplexity on documents stratified by difficulty (centrality).
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        df_corpus: DataFrame with 'text' column
        centrality: Array of centrality values (high=easy, low=hard)
        n_bins: Number of difficulty bins
        n_samples_per_bin: Samples to evaluate per bin
        device: Device to use
    
    Returns:
        Dict with perplexity per difficulty level
    """
    model.eval()
    
    # Create difficulty bins based on centrality percentiles
    percentiles = np.linspace(0, 100, n_bins + 1)
    thresholds = [np.percentile(centrality, p) for p in percentiles]
    
    results = {}
    bin_labels = ['Hard (Low Centrality)', 'Medium', 'Easy (High Centrality)'] if n_bins == 3 else [f'Bin {i}' for i in range(n_bins)]
    
    for bin_idx in range(n_bins):
        # Get documents in this difficulty bin
        low_thresh = thresholds[bin_idx]
        high_thresh = thresholds[bin_idx + 1]
        
        if bin_idx == n_bins - 1:  # Last bin includes upper bound
            mask = (centrality >= low_thresh) & (centrality <= high_thresh)
        else:
            mask = (centrality >= low_thresh) & (centrality < high_thresh)
        
        bin_indices = np.where(mask)[0]
        
        # Sample documents from this bin
        if len(bin_indices) > n_samples_per_bin:
            sample_indices = np.random.choice(bin_indices, n_samples_per_bin, replace=False)
        else:
            sample_indices = bin_indices
        
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for idx in sample_indices:
                text = df_corpus.iloc[idx]['text']
                
                encodings = tokenizer(
                    text,
                    truncation=True,
                    max_length=512,
                    padding='max_length',
                    return_tensors='pt'
                )
                
                input_ids = encodings['input_ids'].to(device)
                attention_mask = encodings['attention_mask'].to(device)
                
                # Compute loss without masking (causal-style for perplexity)
                labels = input_ids.clone()
                
                with autocast(enabled=(device == 'cuda')):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                
                # Count non-padding tokens
                n_tokens = attention_mask.sum().item()
                total_loss += outputs.loss.item() * n_tokens
                total_tokens += n_tokens
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('nan')
        perplexity = np.exp(avg_loss)
        
        results[bin_labels[bin_idx]] = {
            'perplexity': perplexity,
            'avg_loss': avg_loss,
            'n_samples': len(sample_indices),
            'centrality_range': (low_thresh, high_thresh)
        }
    
    return results


def compute_sample_efficiency(trainer, metric='val_loss'):
    """
    Compute sample efficiency metrics.
    
    Returns validation loss vs unique documents seen.
    """
    sampler = trainer.sampler
    
    # Track unique docs over time
    unique_docs_over_time = []
    seen = set()
    for doc in sampler.sampled_docs:
        seen.add(doc)
        unique_docs_over_time.append(len(seen))
    
    # Calculate batches per epoch
    n_batches_per_epoch = len(sampler.sampled_docs) // (len(trainer.epoch_losses) * sampler.batch_size) if len(trainer.epoch_losses) > 0 else 1
    n_batches_per_epoch = max(1, len(sampler.sampled_docs) // max(1, len(trainer.epoch_losses)))
    
    # Get unique docs at end of each epoch
    unique_docs_per_epoch = []
    for epoch in range(len(trainer.epoch_losses)):
        end_idx = min((epoch + 1) * n_batches_per_epoch * sampler.batch_size, len(unique_docs_over_time))
        if end_idx > 0:
            unique_docs_per_epoch.append(unique_docs_over_time[end_idx - 1])
        else:
            unique_docs_per_epoch.append(0)
    
    return {
        'unique_docs_per_epoch': unique_docs_per_epoch,
        'total_unique_docs': len(seen),
        'total_samples': len(sampler.sampled_docs),
        'unique_docs_over_time': unique_docs_over_time
    }


def compute_masked_prediction_accuracy(model, tokenizer, df_corpus, indices, 
                                        mask_prob=0.15, device='cuda', seed=42):
    """
    Compute masked token prediction accuracy.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer  
        df_corpus: DataFrame with 'text' column
        indices: Document indices to evaluate
        mask_prob: Probability of masking each token
        device: Device to use
        seed: Random seed for reproducibility
    
    Returns:
        Dict with accuracy metrics
    """
    model.eval()
    torch.manual_seed(seed)
    
    total_correct = 0
    total_masked = 0
    
    # Track accuracy by token frequency (using simple heuristic)
    frequent_correct = 0
    frequent_total = 0
    rare_correct = 0
    rare_total = 0
    
    # Get vocabulary size for frequency estimation
    vocab_size = tokenizer.vocab_size
    
    with torch.no_grad():
        for idx in tqdm(indices, desc="Computing MLM accuracy"):
            text = df_corpus.iloc[idx]['text']
            
            encodings = tokenizer(
                text,
                truncation=True,
                max_length=512,
                padding='max_length',
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            original_ids = input_ids.clone()
            
            # Create mask (same as training)
            rand = torch.rand(input_ids.shape, device=device)
            mask_arr = (rand < mask_prob) & (input_ids != tokenizer.pad_token_id)
            
            # Store masked positions
            masked_positions = mask_arr.nonzero(as_tuple=True)
            
            if len(masked_positions[0]) == 0:
                continue
            
            # Apply mask
            input_ids[mask_arr] = tokenizer.mask_token_id
            
            # Get predictions
            with autocast(enabled=(device == 'cuda')):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            predictions = outputs.logits.argmax(dim=-1)
            
            # Compare predictions to original tokens at masked positions
            for i in range(len(masked_positions[0])):
                batch_idx = masked_positions[0][i]
                pos_idx = masked_positions[1][i]
                
                predicted_token = predictions[batch_idx, pos_idx].item()
                original_token = original_ids[batch_idx, pos_idx].item()
                
                is_correct = predicted_token == original_token
                total_masked += 1
                if is_correct:
                    total_correct += 1
                
                # Heuristic: tokens with ID < vocab_size/10 are more frequent
                if original_token < vocab_size // 10:
                    frequent_total += 1
                    if is_correct:
                        frequent_correct += 1
                else:
                    rare_total += 1
                    if is_correct:
                        rare_correct += 1
    
    return {
        'overall_accuracy': total_correct / total_masked if total_masked > 0 else 0,
        'total_correct': total_correct,
        'total_masked': total_masked,
        'frequent_token_accuracy': frequent_correct / frequent_total if frequent_total > 0 else 0,
        'rare_token_accuracy': rare_correct / rare_total if rare_total > 0 else 0,
        'n_frequent_tokens': frequent_total,
        'n_rare_tokens': rare_total
    }


def plot_additional_evaluations(curriculum_results, random_results, 
                                 curriculum_efficiency, random_efficiency,
                                 curriculum_perplexity, random_perplexity,
                                 save_path='additional_evaluations.png'):
    """
    Plot additional evaluation metrics.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Perplexity by difficulty
    ax = axes[0, 0]
    difficulties = list(curriculum_perplexity.keys())
    curr_ppl = [curriculum_perplexity[d]['perplexity'] for d in difficulties]
    rand_ppl = [random_perplexity[d]['perplexity'] for d in difficulties]
    
    x = np.arange(len(difficulties))
    width = 0.35
    ax.bar(x - width/2, curr_ppl, width, label='Curriculum', color='blue', alpha=0.7)
    ax.bar(x + width/2, rand_ppl, width, label='Random', color='orange', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(difficulties, rotation=15, ha='right')
    ax.set_ylabel('Perplexity')
    ax.set_title('Perplexity by Document Difficulty')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Sample efficiency - Val loss vs unique docs
    ax = axes[0, 1]
    curr_unique = curriculum_efficiency['unique_docs_per_epoch']
    rand_unique = random_efficiency['unique_docs_per_epoch']
    
    # Need to align with val_losses (skip epoch 0)
    if len(curr_unique) > 1:
        ax.plot(curr_unique[1:], label='Curriculum', marker='o')
        ax.plot(rand_unique[1:], label='Random', marker='s')
    ax.set_xlabel('Unique Documents Seen')
    ax.set_ylabel('Epoch')
    ax.set_title('Document Coverage Over Epochs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: MLM Accuracy comparison
    ax = axes[0, 2]
    metrics = ['Overall', 'Frequent Tokens', 'Rare Tokens']
    curr_acc = [curriculum_results['overall_accuracy'], 
                curriculum_results['frequent_token_accuracy'],
                curriculum_results['rare_token_accuracy']]
    rand_acc = [random_results['overall_accuracy'],
                random_results['frequent_token_accuracy'], 
                random_results['rare_token_accuracy']]
    
    x = np.arange(len(metrics))
    ax.bar(x - width/2, [a*100 for a in curr_acc], width, label='Curriculum', color='blue', alpha=0.7)
    ax.bar(x + width/2, [a*100 for a in rand_acc], width, label='Random', color='orange', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Masked Token Prediction Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Perplexity improvement by difficulty
    ax = axes[1, 0]
    improvements = [(rand_ppl[i] - curr_ppl[i]) / rand_ppl[i] * 100 for i in range(len(difficulties))]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax.bar(difficulties, improvements, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Improvement (%)')
    ax.set_title('Curriculum Perplexity Improvement by Difficulty\n(positive = curriculum better)')
    ax.set_xticklabels(difficulties, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Cumulative unique documents over samples
    ax = axes[1, 1]
    # Subsample for plotting efficiency
    step = max(1, len(curriculum_efficiency['unique_docs_over_time']) // 1000)
    ax.plot(curriculum_efficiency['unique_docs_over_time'][::step], label='Curriculum', alpha=0.7)
    ax.plot(random_efficiency['unique_docs_over_time'][::step], label='Random', alpha=0.7)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Unique Documents Seen')
    ax.set_title('Sample Efficiency: Document Coverage')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Summary statistics table
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_data = [
        ['Metric', 'Curriculum', 'Random', '\u0394 (%)'],
        ['Overall MLM Acc.', f"{curriculum_results['overall_accuracy']*100:.2f}%", 
         f"{random_results['overall_accuracy']*100:.2f}%",
         f"{(curriculum_results['overall_accuracy']-random_results['overall_accuracy'])*100:+.2f}"],
        ['Rare Token Acc.', f"{curriculum_results['rare_token_accuracy']*100:.2f}%",
         f"{random_results['rare_token_accuracy']*100:.2f}%",
         f"{(curriculum_results['rare_token_accuracy']-random_results['rare_token_accuracy'])*100:+.2f}"],
        ['Avg Perplexity', f"{np.mean(curr_ppl):.2f}", f"{np.mean(rand_ppl):.2f}",
         f"{(np.mean(rand_ppl)-np.mean(curr_ppl))/np.mean(rand_ppl)*100:+.2f}"],
        ['Hard Doc PPL', f"{curr_ppl[0]:.2f}", f"{rand_ppl[0]:.2f}",
         f"{(rand_ppl[0]-curr_ppl[0])/rand_ppl[0]*100:+.2f}"],
        ['Unique Docs Seen', f"{curriculum_efficiency['total_unique_docs']}",
         f"{random_efficiency['total_unique_docs']}", '-']
    ]
    
    table = ax.table(cellText=summary_data, loc='center', cellLoc='center',
                     colWidths=[0.3, 0.23, 0.23, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style header row
    for j in range(4):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    ax.set_title('Evaluation Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


#%%
def plot_training_comparison(curriculum_trainer: CurriculumTrainer,
                            random_trainer: RandomTrainer,
                            save_path: str = None):
    """
    Plot comparison of curriculum vs random training.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Training loss over time
    ax = axes[0, 0]
    ax.plot(curriculum_trainer.train_losses, label='Curriculum', alpha=0.7)
    ax.plot(random_trainer.train_losses, label='Random', alpha=0.7)
    ax.set_xlabel('Batch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Epoch average loss (train) - skip epoch 0
    ax = axes[0, 1]
    epochs_train = range(1, len(curriculum_trainer.epoch_losses))
    ax.plot(epochs_train, curriculum_trainer.epoch_losses[1:], marker='o', label='Curriculum Train')
    ax.plot(epochs_train, random_trainer.epoch_losses[1:], marker='s', label='Random Train')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Loss')
    ax.set_title('Epoch Average Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Validation loss comparison - skip epoch 0
    ax = axes[0, 2]
    if curriculum_trainer.val_losses and random_trainer.val_losses:
        epochs_val = range(1, len(curriculum_trainer.val_losses))
        ax.plot(epochs_val, curriculum_trainer.val_losses[1:], marker='o', label='Curriculum Val', color='blue')
        ax.plot(epochs_val, random_trainer.val_losses[1:], marker='s', label='Random Val', color='orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Validation Loss Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No validation data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Validation Loss (No Data)')
    
    # Plot 4: Curriculum batch centrality progression (inverted: low = hard)
    ax = axes[1, 0]
    ax.plot(curriculum_trainer.batch_centralities, alpha=0.7)
    ax.set_xlabel('Batch')
    ax.set_ylabel('Average Batch Centrality')
    ax.set_title('Curriculum: Centrality Progression (↓ = harder docs)')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Document sampling distribution by centrality
    ax = axes[1, 1]
    curriculum_centralities = curriculum_trainer.sampler.sampled_centralities
    ax.hist(curriculum_centralities, bins=50, alpha=0.7, label='Curriculum', density=True)
    ax.axvline(np.mean(curriculum_centralities), color='blue', linestyle='--', 
               label=f'Curriculum Mean: {np.mean(curriculum_centralities):.6f}')
    ax.set_xlabel('Centrality (Stationary Distribution)')
    ax.set_ylabel('Density')
    ax.set_title('Sampled Document Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Train vs Val loss for each method - skip epoch 0
    ax = axes[1, 2]
    if curriculum_trainer.val_losses and random_trainer.val_losses:
        epochs = range(1, len(curriculum_trainer.epoch_losses))
        ax.plot(epochs, curriculum_trainer.epoch_losses[1:], 'b-', marker='o', label='Curriculum Train')
        ax.plot(epochs, curriculum_trainer.val_losses[1:], 'b--', marker='o', label='Curriculum Val')
        ax.plot(epochs, random_trainer.epoch_losses[1:], 'orange', marker='s', label='Random Train')
        ax.plot(epochs, random_trainer.val_losses[1:], color='orange', linestyle='--', marker='s', label='Random Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Train vs Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No validation data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Train vs Val (No Data)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("TRAINING COMPARISON SUMMARY")
    print("="*60)
    print(f"\nCurriculum Training:")
    print(f"  Final Train Loss: {curriculum_trainer.epoch_losses[-1]:.4f}")
    print(f"  Best Train Loss: {min(curriculum_trainer.epoch_losses):.4f}")
    if curriculum_trainer.val_losses:
        print(f"  Final Val Loss: {curriculum_trainer.val_losses[-1]:.4f}")
        print(f"  Best Val Loss: {min(curriculum_trainer.val_losses):.4f}")
    print(f"  Avg Centrality Sampled: {np.mean(curriculum_centralities):.6f}")
    
    print(f"\nRandom Training:")
    print(f"  Final Train Loss: {random_trainer.epoch_losses[-1]:.4f}")
    print(f"  Best Train Loss: {min(random_trainer.epoch_losses):.4f}")
    if random_trainer.val_losses:
        print(f"  Final Val Loss: {random_trainer.val_losses[-1]:.4f}")
        print(f"  Best Val Loss: {min(random_trainer.val_losses):.4f}")
    
    # Compute improvements
    train_improvement = (random_trainer.epoch_losses[-1] - curriculum_trainer.epoch_losses[-1]) / random_trainer.epoch_losses[-1] * 100
    print(f"\nTrain Loss Improvement (Curriculum vs Random): {train_improvement:.2f}%")
    
    if curriculum_trainer.val_losses and random_trainer.val_losses:
        val_improvement = (random_trainer.val_losses[-1] - curriculum_trainer.val_losses[-1]) / random_trainer.val_losses[-1] * 100
        print(f"Val Loss Improvement (Curriculum vs Random): {val_improvement:.2f}%")
    print("="*60)


#%%
# Example usage and main training loop
if __name__ == "__main__":
    # Load data (assumes df_corpus and markov_chain are already computed)
    # This would come from the diffusion_geodesics.py pipeline
    
    print("Loading corpus and computing diffusion geometry...")
    
    # Example parameters
    BATCH_SIZE = 16
    N_EPOCHS = 5
    PATH_DEPTH_MIN = 2
    PATH_DEPTH_MAX = 4
    LEARNING_RATE = 5e-5
    MODEL_NAME = "bert-base-uncased"  # or "distilbert-base-uncased" for faster training
    
    # Load tokenizer and model
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Initialize two separate models for fair comparison
    model_curriculum = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    model_random = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    
    # Initialize samplers
    print("Initializing samplers...")
    curriculum_sampler = CurriculumSampler(
        markov_chain=markov_chain,
        svd_entropy=df_corpus['svd_entropy'].values,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        path_depth_min=PATH_DEPTH_MIN,
        path_depth_max=PATH_DEPTH_MAX,
        curriculum_schedule="cosine"
    )
    
    random_sampler = RandomSampler(
        n_docs=len(df_corpus),
        batch_size=BATCH_SIZE
    )
    
    # Initialize trainers
    print("Initializing trainers...")
    curriculum_trainer = CurriculumTrainer(
        model=model_curriculum,
        tokenizer=tokenizer,
        df_corpus=df_corpus,
        sampler=curriculum_sampler,
        learning_rate=LEARNING_RATE,
        n_epochs=N_EPOCHS
    )
    
    random_trainer = RandomTrainer(
        model=model_random,
        tokenizer=tokenizer,
        df_corpus=df_corpus,
        sampler=random_sampler,
        learning_rate=LEARNING_RATE,
        n_epochs=N_EPOCHS
    )
    
    # Train both models
    print("\n" + "="*60)
    print("CURRICULUM TRAINING")
    print("="*60)
    curriculum_trainer.train()
    
    print("\n" + "="*60)
    print("RANDOM BASELINE TRAINING")
    print("="*60)
    random_trainer.train()
    
    # Plot comparison
    plot_training_comparison(
        curriculum_trainer,
        random_trainer,
        save_path='curriculum_vs_random_training.png'
    )
    
    # Save models
    print("\nSaving models...")
    model_curriculum.save_pretrained("./models/curriculum_model")
    model_random.save_pretrained("./models/random_model")
    
    print("\nTraining complete!")

#%%
