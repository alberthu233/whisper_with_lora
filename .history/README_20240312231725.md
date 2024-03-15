# LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS
A transformer 2024 spring presentation for the paper "Low-Rank Adaptation of Large Language Models" and its implementation.

## Overview
- **Importance of Adapting Large Pre-Trained Language Models to Downstream Tasks**
  - Large pre-trained language models have shown great performance across a wide range of NLP tasks.
  - Adapting these models to specific tasks enhances their utility and application in real-world scenarios.

- **Challenges of Fine-Tuning Large Models**
  - Traditional fine-tuning requires adjustments to all parameters, increasing computational and memory requirements.
  - Maintaining separate model instances for each task is resource-intensive and not scalable.

- **Introduction of Low-Rank Adaptation (LoRA)**
  - LoRA proposes a parameter-efficient adaptation method by introducing trainable rank decomposition matrices into each Transformer layer.
  - This method significantly reduces the number of trainable parameters needed for task adaptation.

- **How LoRA Works and Its Advantages**
  - LoRA freezes the pre-trained model weights, only adjusting the low-rank matrices for task-specific adaptation.
  - It reduces the GPU memory requirements and computational overhead compared to traditional fine-tuning methods.
  - Unlike adapters, LoRA does not introduce additional inference latency, making it efficient for deployment.
  - Empirical results show LoRA's effectiveness in maintaining or improving model performance on downstream tasks with far fewer trainable parameters.

- **Advantages Over Fine-Tuning and Other Adaptation Methods**
  - Memory and storage efficiency: Significantly reduces the VRAM usage and size of model checkpoints.
  - Parameter efficiency: Decreases the number of trainable parameters by orders of magnitude.
  - Flexibility and scalability: Allows for easy adaptation to multiple tasks with minimal additional resource requirements.
  - Performance: Achieves comparable or superior task performance with reduced parameter count and training resources.


