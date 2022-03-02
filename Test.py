import torch
a = torch.tensor([[2],[2],[2],[2],[2]])
b = a ** 2
print(b)
# print(a.size())

# print(a.repeat(1,2).size())
# print(torch.diag(a).unsqueeze(dim=0).size())
# print(torch.diag_embed(a).size())