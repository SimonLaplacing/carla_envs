import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from memory_profiler import profile

################################## set device ##################################
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()


class PathAttention(nn.Module):
    def __init__(self, args):
        super(PathAttention, self).__init__()
        self.args = args
        # actor_init
        self.fc11 = nn.Linear(args.input_dim, args.hidden_dim1)
        self.fc12 = nn.Linear(args.input_dim, args.hidden_dim1)
        self.fc13 = nn.Linear(args.input_dim, args.hidden_dim1)
        self.fc2 = nn.MultiheadAttention(args.hidden_dim1, args.n_heads,batch_first=True)

    def select(self, ego, npc):
        q_ = self.fc11(ego)
        k_ = self.fc12(npc)
        v_ = self.fc13(ego)
        _,weight = self.fc2(q_,k_,v_)
        npc_weight = torch.sum(weight, dim=-2)
        ind = torch.argsort(npc_weight, dim=-1).unsqueeze(-2)
        ind = ind.repeat(1,8,1)
        # print(ind.shape)
        best_npc = ind[:,:,:self.args.select_num]
        ego_weight1 = torch.gather(weight,dim=-1, index=best_npc)
        ego_weight = torch.sum(ego_weight1, dim=-1)
        best_ego = torch.argmax(ego_weight, dim=-1)
        # print(npc_weight.shape,ind.shape,ego_weight1.shape,ego_weight.shape)

        return best_ego, best_npc[:,0,:]

    def lr_decay(self, total_steps):
        lr_now = 0.9 * self.lr * (1 - total_steps / self.max_train_steps) + 0.1 * self.lr
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now

    def save(self, checkpoint_path):
        torch.save(self.ac.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.ac.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

if __name__=="__main__":
    parser = argparse.ArgumentParser("Hyperparameter Setting")
    parser.add_argument("--input_dim", type=int, default=30)
    parser.add_argument("--hidden_dim1", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--select_num", type=int, default=3)
    args = parser.parse_args()
    ego = torch.rand(2,8,30)
    npc = torch.rand(2,8,30)
    path = PathAttention(args)
    best_ego,best_npc = path.select(ego,npc)
    print(best_ego.shape)
    print(best_npc.shape)




