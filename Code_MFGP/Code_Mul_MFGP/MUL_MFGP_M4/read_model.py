from model import MyVGG
import torch


device = torch.device('cuda')
model = torch.load(r'../output_grs/5_241212_23_59_23/best_model.pth.tar').to(device)
for name in model.state_dict():
  print(name)
print(model)

