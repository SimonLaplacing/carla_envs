import torch
a=torch.tensor([[[1],[2],[3]],[[4],[5],[6]],[[7],[8],[9]]])
b=a.view(-1,3,3,1)
c=b.view(3,3,1)
print(a.shape,a)

print(b.shape,b)

print(c.shape,c)
# a(1,2,3,3,3,3,4)

