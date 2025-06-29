# LLM Classification Finetuning Hackathon

**Kaggle | June 2025**

---

## Overview

Achieved a top leaderboard position in a Kaggle hackathon focused  on LLM (Large Language Model) classification. The goal was to fine-tune a custom architecture built on DeBERTa v3 (extra small) using KerasNLP and JAX, to classify which of two LLM responses (or a tie) was preferred for a given prompt.

---

## Key Features

- **Custom Dual-Passage Classifier:** Processes two LLM responses per prompt using shared DeBERTa embeddings, concatenation, and global average pooling for robust classification.
- **Optimized Training Pipeline:** Preprocesses prompt–response pairs, automates tokenization, and handles class label conversion.
- **Advanced Training Techniques:** Uses cosine annealing learning rate scheduling, label smoothing, and balanced evaluation metrics (categorical accuracy, log loss) for improved generalization.
- **Scalable Design:** Ready for ensembling multiple LLMs and can be extended for future LLM evaluation tasks.

---
