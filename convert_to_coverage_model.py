import torch, sys, dill

ckpt_path = sys.argv[1]
new_ckpt_path = sys.argv[2]

ckpt = torch.load(ckpt_path)
# Initialize weights
w = torch.randn(512, 1) * 0.05
# Set coverage weight
ckpt['model']['decoder.attn.linear_cover.weight'] = w
torch.save(ckpt, new_ckpt_path, pickle_module=dill)
