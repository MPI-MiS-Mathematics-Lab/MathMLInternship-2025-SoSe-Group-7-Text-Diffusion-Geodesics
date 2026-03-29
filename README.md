# Knowledge Discovery in Text Corpora with Diffusion Maps and Information Theory

## Abstract

> *Modeling Human Learning as Paths Along Entropy Gradients in Scientific Text Corpora*
> *— Entropy 2026 Conference*

The evolution of scientific language reflects the dynamic nature of human knowledge acquisition, with new concepts emerging at the research frontier characterized by low-probability word combinations carrying high information content. While machine learning frames learning as entropy reduction, human knowledge integration fundamentally involves navigating from general, low-entropy foundational concepts toward specialized, high-entropy advanced material. This work presents a novel information-theoretic framework that models human learning as directed paths along entropy gradients in document spaces, characterized through asymmetric cross-entropy relationships.

We leverage Latent Semantic Analysis (SVD of TF-IDF matrices) to extract latent conceptual dimensions and introduce an entropy-based complexity measure computed on non-negative SVD components. This measure captures document position on a generality-specialization spectrum, characterizing the distribution of information across latent semantic features. Going beyond symmetric similarity metrics, we employ cross-entropy as a directed document similarity measure, interpreting the asymmetry through the lens of encoding one document's content using another's probability distribution — a natural analogue to the directional nature of knowledge prerequisites and concept dependencies.

Analysis of OpenWebMath and Wikipedia corpora is conducted using word rank distributions and cross-entropy relative to corpus-wide word frequencies as complementary complexity measures alongside our SVD-based entropy. High-entropy documents tend to exhibit specialized, narrow terminology distributions, while low-entropy documents employ more general, foundational vocabularies. The asymmetric cross-entropy kernel naturally encodes which documents serve as conceptual prerequisites for others, enabling the construction of entropy-guided learning trajectories that progress from foundational to advanced material.

This framework provides a principled information-theoretic approach to modeling human learning paths, curriculum design, and understanding knowledge organization in scientific discourse. By modeling learning as navigation along entropy gradients — from broadly accessible, low-complexity texts toward specialized, high-complexity material — we bridge concepts from information theory, human cognition, and computational linguistics to offer new perspectives on how learners traverse the landscape of scientific knowledge.

**Keywords:** information theory, cross-entropy, document complexity, entropy measures, latent semantic analysis, knowledge structure

---

## Visualizations

<img width="900" height="810" alt="Geodesic path through document manifold" src="https://github.com/user-attachments/assets/1fc9fddb-9fb7-473a-ab95-e8de92d01db6" />

<img alt="Spring layout colored by diffusion sigma parameter t" src="figures/spring_sigma_t.png" />

---

## Example Geodesic Path

The table below shows a shortest geodesic path through the document manifold, progressing from a low-entropy (general, foundational) document to a high-entropy (specialized, advanced) document. SVD entropy increases monotonically along the path, reflecting the learning trajectory from introductory to expert material.

| Step | Document excerpt | SVD Entropy |
|:----:|:-----------------|:-----------:|
| 0 | Introducing matrices — Here, I will introduce the three main ways of thinking about matrices. This high-level de... | 0.4861 |
| 1 | Data for Exercise 7.40 — Backtoback. A data frame/tibble with 24 observations on two variables: score (numeric)... | 0.5664 |
| 2 | Autoencoder — Difference between generative and discriminative modelling. Generative modelling... | 0.5233 |
| 3 | Measure the uncertainty in deep learning models using dropout — Seminal blog post of Yarin Gal from Cambridge... | 0.6269 |
| 4 | A quick-and-dirty introduction into a neural network architecture type called Autoencoder... | 0.5695 |
| 5 | Conv Layers — Each neuron in the second conv layer is connected only to neurons within a small rectangle... | 0.5963 |
| 6 | Chapter 2: Matrices and Linear Algebra — import igl, scipy, numpy, meshplot... | 0.5848 |
| 7 | A practical introduction to GNNs — Part 1 of an introductory lecture on graph neural networks... | 0.6182 |
| 8 | Generative Adversarial Network (GAN) in TensorFlow — Part 4: The GAN Class and Data Functions... | 0.6188 |
| 9 | Lecture 6: Further Examples of Classifiers — More classifiers and their applications. Support Vector... | 0.6992 |
| 10 | K-Nearest Neighbor from Scratch in Python — We are going to implement K-nearest neighbor... | 0.6884 |
| 11 | Multi-layer Perceptron (MLP) — Everything related to Multi-layer perceptron... | 0.6415 |
| 12 | TensorFlow 101: Word2Vec — TensorFlow is a powerful open source library used for large-scale... | 0.7221 |
| 13 | Libraries — import numpy, pandas, matplotlib, seaborn... | 0.6781 |
| 14 | 2.1 Three Popular Data Displays — Learning to interpret the meaning of three graphical representations... | 0.7286 |
| 15 | Machine learning — A subfield of computer science and artificial intelligence... | 0.7235 |
| 16 | Exploring and Understanding Hyperparameter Tuning — Learners use hyperparameters to achieve better performance... | 0.7651 |
| 17 | Building recurrent neural networks with the TensorFlow API... | 0.7719 |
| 18 | LSTMs to Model Physiological Time Series — Harini Suresh, Nicholas Locascio, MIT... | 0.7738 |
<img alt="spring sigma t" src="figures/spring_sigma_t.png" />
