import torch
from gpt4.gpt4 import GPT4

x = torch.randint(0, 256, (1, 1024)).cuda()

model = GPT4()

model(x)

