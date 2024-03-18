# LoRA: Low-Rank Adaptation of Large Language Models

## Introduction
- Adapting large language models to downstream tasks is useful but challenging
  - Fine-tuning is expensive and resource-intensive
  - Models like GPT-3 (175B parameters) require significant resources
- Existing parameter-efficient adaptation techniques have limitations
  - Adapter layers increase inference latency
  - Prompt-based methods reduce available sequence length
- LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)
  - Efficient adaptation technique
  - Adds low-rank update matrices to pre-trained weights
  - Reduces trainable parameters while maintaining performance
- Presentation Overview
  - Explore LoRA paper in detail
  - Discuss mathematical formulation and implementation
  - Present experimental results comparing LoRA with other adaptation methods, full finetuning, freeze part of model, and LoRA adaptation.


## Background
### Fine-Tuning and Parameter-Efficient Adaptation
- Traditional Fine-Tuning
  - Update all model parameters during adaptation
  - Requires storing a separate copy of the model for each downstream task
  - Expensive for large models like GPT-3

- Parameter-Efficient Adaptation Techniques
  - Adapter Layers (Houlsby et al. 2019) (Lin et al. 2020)
    - Insert additional layers between pre-trained weights
    - Capture task-specific information
    - Increase inference latency due to additional computations
  - Prompt-Based Methods (e.g., Prefix-Tuning, Li and Liang, 2021)
    - Optimize continuous prompts to steer model's behavior
    - Do not modify model weights
    - Reduce available sequence length for input tokens
  - Limitations of Existing Techniques
    - Increased inference latency (Adapter Layers)
    - Reduced sequence length (Prompt-Based Methods)
    - Need for more efficient adaptation methods

## LoRA: Low-Rank Adaptation

Breakdown steps of LoRA:
- Initial Weight Matrix: $W_0 \in \mathbb{R}^{d \times k}$ 
- Update to the weight matrix: $\Delta W$ is represented as the product of two matrices $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$
- Low-Rank Constraint: The rank r is much smaller than both $d$ and $k$, where $r \ll \min(d, k)$
- Applying Update: adding low-rank update matrices to the pre-trained weights $W_0 + \Delta W = W_0 + BA$
- Compute the Final output: $h = W_0x + \Delta Wx = W_0x + BAx$

Here is how lora looks like for an actual matrix:
![LoRA Diagram](images/01.png)

During the LoRa initilization, the paper use a random Gaussian initialization for A and zero for B, so $\Delta W = BA$ is zero at the beginning of training. 

We then scale $\Delta Wx$ by $\frac{\alpha}{r}$, where $\alpha$ is a constant in $\mathbb{R}$. 

When optimizing with Adam, tuning \alpha is roughly the same as tuning the learning rate if we scale the initialization appropriately. As a result, we simply set \alpha to the first \sqrt{r} we try and do not tune it. This scaling helps to reduce the need to retune hyperparameters when we vary r （yang 2021）.



- Describe the mathematical formulation of LoRA and how it modifies the attention layers
- Highlight the advantages of LoRA, such as reduced training cost and inference latency

## Code Demo and Results
### Experimental Setup
- Dataset: Gravitation Wave Detection
- Model: Whisper Encoder(Transformer-based model) + Classification Head
- Adaptation Techniques:
  - Full Fine-Tuning
  - Freeze Encoder Part of Model
  - LoRA

### Implementation Details
- Step to implementation of LoRA from scratch in PyTorch
  - Use `torch.nn.Parameter` to define low-rank update matrices A and B
  - Implement LoRA linear layer with low-rank update in the forward pass
  - Replace the original linear layer in the Whisper model with LoRA layer

- Code Snippet for Implementing LoRA Layer
  - See full code in the accompanying Jupyter notebook and src folder

```python
class LoRA_layer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank))
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
        self.rank = rank

    def forward(self, x):
        # @ is matrix multiplication
        x = self.alpha / self.rank * (x @ self.A @ self.B)
        return x

class LoRa_linear(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRA_layer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)

# Replace the original linear layer with LoRA layer
for layer in model.encoder.blocks:
    if args.lora_q:
        layer.attn.query = replace_lora(layer.attn.query)
    if args.lora_k:
        layer.attn.key = replace_lora(layer.attn.key)
    if args.lora_v:
        layer.attn.value = replace_lora(layer.attn.value)
```

When we print the model, the attention layer looks like this:
```
....
(blocks): ModuleList(
      (0-11): 12 x ResidualAttentionBlock(
        (attn): MultiHeadAttention(
          (query): Linear(in_features=768, out_features=768, bias=True)
          (key): Linear(in_features=768, out_features=768, bias=False)
          (value): Linear(in_features=768, out_features=768, bias=True)
          (out): Linear(in_features=768, out_features=768, bias=True)
        )
....
```

Once the LoRA layer is replaced, the `print(model)` will show the LoRA layer in the model like this:
```
....
(blocks): ModuleList(
  (0-11): 12 x ResidualAttentionBlock(
    (attn): MultiHeadAttention(
      (query): LoRa_linear(
        (linear): Linear(in_features=768, out_features=768, bias=True)
        (lora): LoRA_layer()
      )
....
```


### Results and Analysis

Compare the trainable parameters and GPU memory usage of the three models
- Full Fine-Tuning (**89M** parameters)
- Freeze Encoder Part of Model (**2.32M** parameters)
- LoRA (**3.4M** parameters at r = 8)

In the actuall training task when the batch size is set to 1, the model with full parameters fine-tuning will take 8.5 GB of gpu memory, the model with encoder part freezed now take 2.2 GB of gpu memory, and the model with encoder part freezed and applied lora linear layer now take 6.1 GB of gpu memory.

![Training time vs. Validation AUC](images/05.png)
*Figure: Training time vs. Validation AUC.*

- results of the three adaptation techniques
  - full_finetune best val_auc: 0.9964, best val_loss: 0.071
  - lora best val_auc: 0.9968, best val_loss: 0.060
  - frozen_encoder best val_auc: 0.9636, best val_loss: 0.232
- Discuss any observations or insights gained from the experiments

## Conclusion
- LoRA is an efficient and effective adaptation technique for large language models
  - Reduces trainable parameters and maintains performance
  - Enables quick task-switching and efficient deployment
- Low-rank structure provides insights into adaptation process
  - Task-specific directions are amplified with low-rank updates
- LoRA opens up new possibilities for efficient adaptation of large models

## References
Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-Efficient Transfer Learning for NLP. arXiv:1902.00751 [cs, stat], June 2019. URL http://arxiv.org/abs/1902. 00751.

Greg Yang and Edward J. Hu. Feature Learning in Infinite-Width Neural Networks.arXiv:2011.14522 [cond-mat], May 2021. URL http://arxiv.org/abs/2011.14522. arXiv: 2011.14522.



## Appendix
- Include any additional details or experiments that couldn't fit in the main presentation
- Provide links to the accompanying Jupyter notebook or other resources