#%%
"""
Integration script to run curriculum learning experiment.
This connects the diffusion geodesics pipeline with the curriculum training system.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.parsing.preprocessing import preprocess_string
import scipy
from tqdm import tqdm
import torch
import warnings
warnings.filterwarnings('ignore')

# Import curriculum training components
from lm_curriculum import (
    CurriculumSampler, RandomSampler,
    CurriculumTrainer, RandomTrainer,
    plot_training_comparison,
    create_train_val_split,
    compute_perplexity_by_difficulty,
    compute_sample_efficiency,
    compute_masked_prediction_accuracy,
    plot_additional_evaluations
)

from transformers import AutoTokenizer, AutoModelForMaskedLM

#%%
# Step 1: Load and preprocess corpus
print("="*60)
print("STEP 1: Loading corpus")
print("="*60)

corpus_file = "../data/wiki_ml_zeroshot.parquet"
df_corpus = pd.read_parquet(corpus_file)

# Filter for minimum length documents
min_length = 1000  # Shorter for faster training
df_corpus = df_corpus.loc[
    (~df_corpus["url"].duplicated(keep="first")) & 
    (df_corpus["text"].str.len() >= min_length)
].reset_index(drop=True)

# For faster experimentation, take a subset
SUBSET_SIZE = 10000  # Adjust based on available resources
if len(df_corpus) > SUBSET_SIZE:
    df_corpus = df_corpus.sample(n=SUBSET_SIZE, random_state=42).reset_index(drop=True)

print(f"Corpus size: {len(df_corpus)} documents")
print(f"Average text length: {df_corpus['text'].str.len().mean():.0f} characters")

#%%
# Step 2: Compute TF-IDF and SVD
print("\n" + "="*60)
print("STEP 2: Computing TF-IDF and SVD")
print("="*60)

vectorizer = TfidfVectorizer(
    max_df=0.5, 
    min_df=5,  # Match diffusion_geodesics.py
    preprocessor=preprocess_string,
    tokenizer=lambda x: x
    # No max_features limit to match diffusion_geodesics.py
)
tfidf_matrix = vectorizer.fit_transform(df_corpus["text"])
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

# SVD dimensionality reduction - use more components like diffusion_geodesics.py
n_markov_components = 5000
n_components = min(n_markov_components, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
svd = TruncatedSVD(n_components=n_components, random_state=42)
svd_matrix = svd.fit_transform(tfidf_matrix)
print(f"SVD matrix shape: {svd_matrix.shape}")

#%%
# Step 3: Create non-negative matrix and compute entropy
print("\n" + "="*60)
print("STEP 3: Computing SVD entropy")
print("="*60)

svd_matrix_nonneg = np.zeros_like(svd_matrix)
epsilon = 1e-5

for i in range(svd_matrix.shape[1]):
    column = svd_matrix[:, i]
    pos_norm = np.linalg.norm(column[column > 0])
    neg_norm = np.linalg.norm(column[column < 0])
    
    if pos_norm >= neg_norm:
        svd_matrix_nonneg[:, i] = np.maximum(column, epsilon)
    else:
        svd_matrix_nonneg[:, i] = np.maximum(-column, epsilon)

# Normalize and compute entropy kernel
svd_matrix_normalized = svd_matrix_nonneg / np.sum(svd_matrix_nonneg, axis=0)
kernel_matrix = - svd_matrix_normalized @ np.log2(svd_matrix_normalized.T)
df_corpus["svd_entropy"] = np.diag(kernel_matrix) / np.log2(kernel_matrix.shape[0])

print(f"Entropy range: [{df_corpus['svd_entropy'].min():.4f}, {df_corpus['svd_entropy'].max():.4f}]")
print(f"Mean entropy: {df_corpus['svd_entropy'].mean():.4f}")

# Visualize entropy distribution
plt.figure(figsize=(10, 6))
plt.hist(df_corpus["svd_entropy"], bins=50, alpha=0.7, edgecolor='black')
plt.xlabel('SVD Entropy')
plt.ylabel('Frequency')
plt.title('Histogram of SVD Entropy')
plt.grid(True, alpha=0.3)
plt.savefig('entropy_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

#%%
# Step 4: Compute Markov chain with eigendecomposition and diffusion
print("\n" + "="*60)
print("STEP 4: Computing Markov chain with diffusion")
print("="*60)

sigma = 1.2  # Kernel bandwidth parameter
similarity_matrix = np.exp(-kernel_matrix / sigma**2)
similarity_matrix -= np.diag(np.diag(similarity_matrix))

print(f"Similarity matrix range: [{similarity_matrix.min():.6f}, {similarity_matrix.max():.6f}]")

# Create base Markov matrix
markov_base = similarity_matrix / (similarity_matrix.sum(axis=0) + 1e-10)

print(f"Base Markov chain shape: {markov_base.shape}")

# Eigendecomposition of Markov chain
print("Computing eigendecomposition...")
eigenvalues, eigenvectors = np.linalg.eig(markov_base.T)
eigenvalues = np.real(eigenvalues)
eigenvectors = np.real(eigenvectors)

# Sort by absolute eigenvalue (descending)
sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

print(f"Top 5 eigenvalues: {eigenvalues[:5]}")

# Compute stationary distribution (first eigenvector, normalized)
stationary_distribution = np.abs(eigenvectors[:, 0])
stationary_distribution = stationary_distribution / stationary_distribution.sum()
df_corpus["centrality"] = stationary_distribution

print(f"Centrality range: [{stationary_distribution.min():.8f}, {stationary_distribution.max():.8f}]")
print(f"Mean centrality: {stationary_distribution.mean():.8f}")

# Apply t=1.2 diffusion steps using eigendecomposition
t = 1.2
print(f"\nApplying t={t} diffusion steps...")
eigv_inv = scipy.linalg.pinv(eigenvectors)

# Compute eigenvalue^t (handle potential complex numbers from fractional powers)
n_components = min(len(eigenvalues), markov_base.shape[0])
eigv_power = np.diag(np.real(np.complex128(eigenvalues[:n_components])**t))

# Reconstruct diffused Markov matrix: M^t = V * D^t * V^{-1}
markov_chain = eigenvectors[:, :n_components] @ eigv_power @ eigv_inv[:n_components, :]
markov_chain = np.real(markov_chain)

# Ensure valid probability matrix (non-negative, columns sum to 1)
markov_chain = np.maximum(markov_chain, 0)
markov_chain = markov_chain / (markov_chain.sum(axis=0, keepdims=True) + 1e-10)

print(f"Diffused Markov chain shape: {markov_chain.shape}")
print(f"Markov chain sum per column (should be ~1): {markov_chain.sum(axis=0).mean():.6f}")

# Visualize centrality distribution
plt.figure(figsize=(10, 6))
plt.hist(np.log10(stationary_distribution + 1e-12), bins=50, alpha=0.7, edgecolor='black')
plt.xlabel('Log10(Centrality)')
plt.ylabel('Frequency')
plt.title('Histogram of Document Centrality (Stationary Distribution)')
plt.grid(True, alpha=0.3)
plt.savefig('centrality_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

#%%
# Step 5: Visualize curriculum progression
print("\n" + "="*60)
print("STEP 5: Visualizing curriculum strategy")
print("="*60)

# Create sampler to visualize the curriculum (centrality-based with Beta distribution)
test_sampler = CurriculumSampler(
    markov_chain=markov_chain,
    stationary_distribution=stationary_distribution,
    n_epochs=10,
    batch_size=32,
    path_depth_min=4,
    path_depth_max=15,
    curriculum_schedule="cosine",
    hard_emphasis=0.7,
    initial_concentration=5.0,
    final_concentration=2.0
)

# Visualize Beta distribution progression
print("Visualizing curriculum sampling distribution...")
test_sampler.visualize_sampling_distribution(epochs_to_show=[0, 2, 5, 7, 9])

# Detailed Beta distribution evolution visualization
from scipy import stats

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Parameters for visualization
n_epochs_viz = 10
x = np.linspace(0, 1, 500)

# Top row: Show PDF at different epochs with filled areas
epochs_to_show = [0, 2, 4, 6, 8, 9]
colors = plt.cm.viridis(np.linspace(0, 1, len(epochs_to_show)))

for idx, (epoch, color) in enumerate(zip(epochs_to_show, colors)):
    ax = axes[idx // 3, idx % 3]
    alpha, beta_param = test_sampler.get_sampling_distribution_params(epoch)
    y = stats.beta.pdf(x, alpha, beta_param)
    
    # Fill under curve
    ax.fill_between(x, y, alpha=0.4, color=color)
    ax.plot(x, y, color=color, linewidth=2)
    
    # Add mean and mode lines
    mean = alpha / (alpha + beta_param)
    if alpha > 1 and beta_param > 1:
        mode = (alpha - 1) / (alpha + beta_param - 2)
    else:
        mode = 0 if alpha < 1 else 1 if beta_param < 1 else 0.5
    
    ax.axvline(mean, color='red', linestyle='--', alpha=0.8, label=f'Mean={mean:.2f}')
    ax.axvline(mode, color='blue', linestyle=':', alpha=0.8, label=f'Mode={mode:.2f}')
    
    ax.set_xlim(0, 1)
    ax.set_xlabel('Difficulty Rank\n(0=Easy, 1=Hard)')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'Epoch {epoch}\nα={alpha:.2f}, β={beta_param:.2f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Beta Distribution Evolution Over Curriculum Training', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('beta_distribution_evolution.png', dpi=300, bbox_inches='tight')
plt.show()

# Animated-style heatmap of PDF evolution
fig, ax = plt.subplots(figsize=(12, 6))

# Create heatmap data
n_epochs_heatmap = 50
pdf_matrix = np.zeros((n_epochs_heatmap, len(x)))

for epoch in range(n_epochs_heatmap):
    # Scale epoch to match test_sampler's n_epochs
    scaled_epoch = epoch * test_sampler.n_epochs / n_epochs_heatmap
    alpha, beta_param = test_sampler.get_sampling_distribution_params(int(scaled_epoch))
    pdf_matrix[epoch, :] = stats.beta.pdf(x, alpha, beta_param)

# Normalize each row for better visualization
pdf_matrix_norm = pdf_matrix / pdf_matrix.max(axis=1, keepdims=True)

im = ax.imshow(pdf_matrix_norm, aspect='auto', cmap='viridis', 
               extent=[0, 1, n_epochs_heatmap, 0], interpolation='bilinear')
ax.set_xlabel('Difficulty Rank (0=Easy, 1=Hard)', fontsize=12)
ax.set_ylabel('Epoch', fontsize=12)
ax.set_title('Curriculum Sampling Distribution Evolution\n(Brighter = Higher Probability)', fontsize=14)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Normalized Probability Density')

# Mark the mean trajectory
means = []
for epoch in range(n_epochs_heatmap):
    scaled_epoch = epoch * test_sampler.n_epochs / n_epochs_heatmap
    alpha, beta_param = test_sampler.get_sampling_distribution_params(int(scaled_epoch))
    means.append(alpha / (alpha + beta_param))

ax.plot(means, range(n_epochs_heatmap), 'r--', linewidth=2, label='Distribution Mean')
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig('beta_distribution_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

print("Beta distribution visualizations saved!")

# Also show median centrality threshold progression for comparison
epoch_thresholds = []
for epoch in range(10):
    threshold = test_sampler.get_centrality_threshold(epoch)
    epoch_thresholds.append(threshold)

plt.figure(figsize=(10, 6))
plt.plot(range(10), epoch_thresholds, marker='o', linewidth=2)
plt.axhline(y=stationary_distribution.mean(), color='r', linestyle='--', 
           label=f'Mean Centrality: {stationary_distribution.mean():.6f}')
plt.xlabel('Epoch')
plt.ylabel('Median Sampling Centrality')
plt.title('Curriculum Learning: Distribution Median Progression\n(Beta distribution shifts toward hard/peripheral docs)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('curriculum_progression.png', dpi=300, bbox_inches='tight')
plt.show()

#%%
# Step 6: Initialize models and tokenizer
print("\n" + "="*60)
print("STEP 6: Loading models")
print("="*60)

MODEL_NAME = "distilbert-base-uncased"  # Faster than BERT for experimentation
print(f"Loading model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model_curriculum = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
model_random = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

print(f"Model parameters: {sum(p.numel() for p in model_curriculum.parameters()):,}")

#%%
# Step 7: Configure training parameters
print("\n" + "="*60)
print("STEP 7: Configuring training")
print("="*60)

# Training configuration
BATCH_SIZE = 32  # Smaller batch for experimentation
N_EPOCHS = 50  # Fewer epochs for faster results
PATH_DEPTH_MIN = 5
PATH_DEPTH_MAX = 10
LEARNING_RATE = 1e-4 # 5e-5
CURRICULUM_SCHEDULE = "cosine"  # Options: "linear", "exponential", "cosine"
HARD_EMPHASIS = 0.7  # Controls how far distribution shifts toward hard docs
INITIAL_CONCENTRATION = 5.0  # Beta distribution concentration at start (higher = more peaked on easy)
FINAL_CONCENTRATION = 2.0  # Beta distribution concentration at end (lower = more spread out)
VAL_RATIO = 0.1  # 10% of data for validation

print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {N_EPOCHS}")
print(f"Path depth: [{PATH_DEPTH_MIN}, {PATH_DEPTH_MAX}]")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Curriculum schedule: {CURRICULUM_SCHEDULE}")
print(f"Hard emphasis: {HARD_EMPHASIS}")
print(f"Beta concentration: {INITIAL_CONCENTRATION} → {FINAL_CONCENTRATION}")
print(f"Validation ratio: {VAL_RATIO}")

# Create train/validation split
train_indices, val_indices = create_train_val_split(len(df_corpus), val_ratio=VAL_RATIO, seed=None)
print(f"Train set: {len(train_indices)} documents")
print(f"Validation set: {len(val_indices)} documents")

# Initialize samplers (centrality-based curriculum with Beta distribution)
curriculum_sampler = CurriculumSampler(
    markov_chain=markov_chain,
    stationary_distribution=stationary_distribution,
    n_epochs=N_EPOCHS,
    batch_size=BATCH_SIZE,
    path_depth_min=PATH_DEPTH_MIN,
    path_depth_max=PATH_DEPTH_MAX,
    curriculum_schedule=CURRICULUM_SCHEDULE,
    hard_emphasis=HARD_EMPHASIS,
    initial_concentration=INITIAL_CONCENTRATION,
    final_concentration=FINAL_CONCENTRATION
)

random_sampler = RandomSampler(
    n_docs=len(df_corpus),
    batch_size=BATCH_SIZE
)

#%%
# Step 8: Train curriculum model
print("\n" + "="*60)
print("STEP 8: Training curriculum model")
print("="*60)

curriculum_trainer = CurriculumTrainer(
    model=model_curriculum,
    tokenizer=tokenizer,
    df_corpus=df_corpus,
    sampler=curriculum_sampler,
    val_indices=val_indices,
    learning_rate=LEARNING_RATE,
    n_epochs=N_EPOCHS
)

curriculum_trainer.train()

#%%
# Step 9: Train random baseline model
print("\n" + "="*60)
print("STEP 9: Training random baseline model")
print("="*60)

random_trainer = RandomTrainer(
    model=model_random,
    tokenizer=tokenizer,
    df_corpus=df_corpus,
    sampler=random_sampler,
    val_indices=val_indices,
    learning_rate=LEARNING_RATE,
    n_epochs=N_EPOCHS
)

random_trainer.train()

#%%
# Step 10: Compare results
print("\n" + "="*60)
print("STEP 10: Comparing results")
print("="*60)

plot_training_comparison(
    curriculum_trainer,
    random_trainer,
    save_path='curriculum_vs_random_training.png'
)

#%%
# Step 11: Analyze sampling patterns
print("\n" + "="*60)
print("STEP 11: Analyzing sampling patterns")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Curriculum document coverage over time
ax = axes[0, 0]
unique_docs_over_time = []
seen_docs = set()
for doc in curriculum_sampler.sampled_docs:
    seen_docs.add(doc)
    unique_docs_over_time.append(len(seen_docs))
ax.plot(unique_docs_over_time)
ax.set_xlabel('Sample')
ax.set_ylabel('Unique Documents Seen')
ax.set_title('Curriculum: Document Coverage Over Time')
ax.grid(True, alpha=0.3)

# Plot 2: Random document coverage over time
ax = axes[0, 1]
unique_docs_over_time_random = []
seen_docs_random = set()
for doc in random_sampler.sampled_docs:
    seen_docs_random.add(doc)
    unique_docs_over_time_random.append(len(seen_docs_random))
ax.plot(unique_docs_over_time_random)
ax.set_xlabel('Sample')
ax.set_ylabel('Unique Documents Seen')
ax.set_title('Random: Document Coverage Over Time')
ax.grid(True, alpha=0.3)

# Plot 3: Centrality progression in curriculum (low = hard)
ax = axes[1, 0]
window_size = 50
moving_avg = pd.Series(curriculum_sampler.sampled_centralities).rolling(window=window_size).mean()
ax.plot(moving_avg, label='Moving Average', linewidth=2)
ax.axhline(y=stationary_distribution.mean(), color='r', linestyle='--', 
          label='Corpus Mean')
ax.set_xlabel('Sample')
ax.set_ylabel('Average Centrality')
ax.set_title(f'Curriculum: Centrality Progression (window={window_size})\n(↓ = harder documents)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Document resampling frequency
ax = axes[1, 1]
doc_counts = pd.Series(curriculum_sampler.sampled_docs).value_counts()
ax.hist(doc_counts.values, bins=30, alpha=0.7, edgecolor='black')
ax.set_xlabel('Times Sampled')
ax.set_ylabel('Number of Documents')
ax.set_title('Curriculum: Document Resampling Frequency')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sampling_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nCurriculum - Unique documents seen: {len(seen_docs)} / {len(df_corpus)} ({len(seen_docs)/len(df_corpus)*100:.1f}%)")
print(f"Random - Unique documents seen: {len(seen_docs_random)} / {len(df_corpus)} ({len(seen_docs_random)/len(df_corpus)*100:.1f}%)")

#%%
# Step 12: Save results
print("\n" + "="*60)
print("STEP 12: Saving results")
print("="*60)

# Save models
model_curriculum.save_pretrained("./models/curriculum_model")
model_random.save_pretrained("./models/random_model")
print("Models saved to ./models/")

# Save training metrics
results = {
    'curriculum': {
        'epoch_losses': curriculum_trainer.epoch_losses,
        'train_losses': curriculum_trainer.train_losses,
        'val_losses': curriculum_trainer.val_losses,
        'batch_centralities': curriculum_trainer.batch_centralities,
        'final_train_loss': curriculum_trainer.epoch_losses[-1],
        'best_train_loss': min(curriculum_trainer.epoch_losses),
        'final_val_loss': curriculum_trainer.val_losses[-1] if curriculum_trainer.val_losses else None,
        'best_val_loss': min(curriculum_trainer.val_losses) if curriculum_trainer.val_losses else None
    },
    'random': {
        'epoch_losses': random_trainer.epoch_losses,
        'train_losses': random_trainer.train_losses,
        'val_losses': random_trainer.val_losses,
        'final_train_loss': random_trainer.epoch_losses[-1],
        'best_train_loss': min(random_trainer.epoch_losses),
        'final_val_loss': random_trainer.val_losses[-1] if random_trainer.val_losses else None,
        'best_val_loss': min(random_trainer.val_losses) if random_trainer.val_losses else None
    },
    'config': {
        'batch_size': BATCH_SIZE,
        'n_epochs': N_EPOCHS,
        'path_depth_min': PATH_DEPTH_MIN,
        'path_depth_max': PATH_DEPTH_MAX,
        'learning_rate': LEARNING_RATE,
        'curriculum_schedule': CURRICULUM_SCHEDULE,
        'hard_emphasis': HARD_EMPHASIS,
        'diffusion_time': t,
        'sigma': sigma,
        'val_ratio': VAL_RATIO,
        'train_size': len(train_indices),
        'val_size': len(val_indices),
        'corpus_size': len(df_corpus)
    }
}

import json
with open('training_results.json', 'w') as f:
    json.dump({k: {k2: v2 if not isinstance(v2, list) else len(v2) for k2, v2 in v.items()} 
               if isinstance(v, dict) else v for k, v in results.items()}, f, indent=2)

print("Results saved to training_results.json")

#%%
# Step 13: Additional Evaluations
print("\n" + "="*60)
print("STEP 13: Additional Evaluations")
print("="*60)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 13a: Compute perplexity by document difficulty (stratified by centrality)
print("\nComputing perplexity by difficulty...")
curriculum_perplexity = compute_perplexity_by_difficulty(
    model_curriculum, tokenizer, df_corpus, stationary_distribution,
    n_bins=3, n_samples_per_bin=100, device=device
)
random_perplexity = compute_perplexity_by_difficulty(
    model_random, tokenizer, df_corpus, stationary_distribution,
    n_bins=3, n_samples_per_bin=100, device=device
)

print("\nPerplexity by Difficulty:")
for difficulty in curriculum_perplexity:
    curr_ppl = curriculum_perplexity[difficulty]['perplexity']
    rand_ppl = random_perplexity[difficulty]['perplexity']
    improvement = (rand_ppl - curr_ppl) / rand_ppl * 100
    print(f"  {difficulty}:")
    print(f"    Curriculum: {curr_ppl:.2f}, Random: {rand_ppl:.2f}, Improvement: {improvement:+.2f}%")

# 13b: Sample efficiency analysis
print("\nComputing sample efficiency...")
curriculum_efficiency = compute_sample_efficiency(curriculum_trainer)
random_efficiency = compute_sample_efficiency(random_trainer)

print(f"\nSample Efficiency:")
print(f"  Curriculum - Unique docs: {curriculum_efficiency['total_unique_docs']}, Total samples: {curriculum_efficiency['total_samples']}")
print(f"  Random - Unique docs: {random_efficiency['total_unique_docs']}, Total samples: {random_efficiency['total_samples']}")

# 13c: Masked token prediction accuracy
print("\nComputing masked token prediction accuracy...")
# Use validation set for evaluation
eval_indices = val_indices[:200]  # Limit for speed

curriculum_mlm_results = compute_masked_prediction_accuracy(
    model_curriculum, tokenizer, df_corpus, eval_indices, device=device
)
random_mlm_results = compute_masked_prediction_accuracy(
    model_random, tokenizer, df_corpus, eval_indices, device=device
)

print(f"\nMasked Token Prediction Accuracy:")
print(f"  Curriculum:")
print(f"    Overall: {curriculum_mlm_results['overall_accuracy']*100:.2f}%")
print(f"    Frequent tokens: {curriculum_mlm_results['frequent_token_accuracy']*100:.2f}%")
print(f"    Rare tokens: {curriculum_mlm_results['rare_token_accuracy']*100:.2f}%")
print(f"  Random:")
print(f"    Overall: {random_mlm_results['overall_accuracy']*100:.2f}%")
print(f"    Frequent tokens: {random_mlm_results['frequent_token_accuracy']*100:.2f}%")
print(f"    Rare tokens: {random_mlm_results['rare_token_accuracy']*100:.2f}%")

# Plot additional evaluations
plot_additional_evaluations(
    curriculum_mlm_results, random_mlm_results,
    curriculum_efficiency, random_efficiency,
    curriculum_perplexity, random_perplexity,
    save_path='additional_evaluations.png'
)

# Update results with additional metrics
results['additional_evaluations'] = {
    'perplexity_by_difficulty': {
        'curriculum': {k: v['perplexity'] for k, v in curriculum_perplexity.items()},
        'random': {k: v['perplexity'] for k, v in random_perplexity.items()}
    },
    'sample_efficiency': {
        'curriculum_unique_docs': curriculum_efficiency['total_unique_docs'],
        'random_unique_docs': random_efficiency['total_unique_docs']
    },
    'mlm_accuracy': {
        'curriculum': curriculum_mlm_results,
        'random': random_mlm_results
    }
}

# Save updated results
with open('training_results_full.json', 'w') as f:
    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    json.dump(convert_to_serializable(results), f, indent=2)

print("\nAdditional evaluation results saved to training_results_full.json")
print("\nExperiment complete!")

#%%
