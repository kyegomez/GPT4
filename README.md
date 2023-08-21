[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)



# GPT4
The open source implementation of the base model behind GPT-4 from OPENAI [Language + Multi-Modal]


# Installation
`pip install gpt4-torch`


# Usage

Here's an illustrative code snippet that showcases GPT-3 in action:


```python
import torch
from gpt4 import GPT4

# Generate a random input sequence
x = torch.randint(0, 256, (1, 1024)).cuda()

# Initialize GPT-3 model
model = GPT4()

# Pass the input sequence through the model
output = model(x)
```


### 📚 Training

```python
from gpt4 import train

train()

```

For further instructions, refer to the [Training SOP](DOCs/TRAINING.md).


1. Set the environment variables:
   - `ENTITY_NAME`: Your wandb project name
   - `OUTPUT_DIR`: Directory to save the weights (e.g., `./weights`)
   - `MASTER_ADDR`: For distributed training
   - `MASTER_PORT` For master port distributed training
   - `RANK`- Number of nodes services
   - `WORLD_SIZE` Number of gpus

2. Configure the training:
   - Accelerate Config
   - Enable Deepspeed 3
   - Accelerate launch train_distributed_accelerate.py

For more information, refer to the [Training SOP](DOCs/TRAINING.md).
