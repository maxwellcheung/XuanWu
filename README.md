### 🧬 XuanWu
Predicting changes in protein thermostability upon mutations using a multiscale geometric graph integrated with a protein language model.

XuanWu is a computational framework that combines a pretrained protein language model (PLM) with a multiscale geometric graph neural network to accurately predict the change in thermostability (ΔΔG) caused by single or multiple mutations in a protein sequence.

### 🔬 Overview
Protein thermostability is critical for both understanding protein function and engineering robust biocatalysts. XuanWU leverages:

Protein Language Model (PLM) – captures evolutionary and sequence-level information.
Multiscale Geometric Graph – models the local and global structural context around mutation sites.
Fusion Architecture – integrates sequence and structural representations for high-accuracy ΔΔG prediction.
### 🚀 Online Server
We recommend using the online web server for easy access without local setup:

👉 https://lujialab.org.cn/server/xuanwu

The server provides a user-friendly interface for submitting mutation data and retrieving predictions.

### 🛠 Local Installation
To set up XuanWu locally, follow these steps:

Prerequisites
conda (recommended for environment management)

```conda env create -f environment.yml```
