import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch import nn
import torchvision.transforms as transforms

device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()

class BEV_handle(nn.Module):
    def __init__(self):
        super(BEV_handle, self).__init__()
    
        self.Norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.BEV_layer = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    def act(self, p):
        p = torch.as_tensor(p, dtype=torch.float)
        p = p.view(-1,p.shape[-3],p.shape[-2],p.shape[-1])
        p = self.Norm(p/255)
        p =self.BEV_layer(p)

        p.detach_()
        return p.cpu().numpy()