import torch
from gpt4.gpt4 import GPT4MultiModal

#usage
img = torch.randn(1, 3, 256, 256)
caption = torch.randint(0, 20000, (1, 1024))

model = GPT4MultiModal()
output = model(img, caption)
print(output.shape) # (1, 1024, 20000)

