from numpy import cov
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torch.nn.functional as F

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init=0.6):
        super(Actor, self).__init__()
        self.action_std_init = action_std_init

        self.fc1 = nn.Linear(state_dim[0], 64)
        self.fc2 = nn.Linear(64,16)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3,stride=1,padding=1),                              
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2))
        self.fc3 = nn.Linear(state_dim[1][0]*state_dim[1][1]*2,1024)
        self.fc4 = nn.Linear(1024,48)

        self.mu_head = nn.Linear(64, action_dim)
        self.sigma_head = nn.Linear(64, 1)

    def forward(self, x1, x2):
        x1 = F.leaky_relu(self.fc1(x1))
        x1 = F.leaky_relu(self.fc2(x1))
        x2 = self.conv1(x2)
        x2 = F.relu(self.fc3(x2.view(x2.size(0),-1)))
        x2 = F.relu(self.fc4(x2))
        x3 = torch.cat((x1,x2[0]),-1)
        mu = F.softsign(self.mu_head(x3))
        sigma = self.action_std_init*torch.sigmoid(self.sigma_head(x3))

        return mu, sigma

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim[0], 64)
        self.fc2 = nn.Linear(64,16)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3,stride=1,padding=2),                              
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2))
        self.fc3 = nn.Linear(state_dim[1][0]*state_dim[1][1]//4,1024)
        self.fc4 = nn.Linear(1024,48)

        self.state_value= nn.Linear(64, 1)

    def forward(self, x1,x2):
        x1 = F.leaky_relu(self.fc1(x1))
        x1 = F.leaky_relu(self.fc2(x1))
        x2 = self.conv1(x2)
        x2 = F.relu(self.fc3(x2.view(x2.size(0),-1)))
        x2 = F.relu(self.fc4(x2))
        x3 = torch.cat((x1,x2[0]),-1)
        value = self.state_value(x3)
        return value

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        if has_continuous_action_space:
            self.action_dim = action_dim
            # self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space :
            self.actor = Actor(state_dim, action_dim, action_std_init)
            # self.actor = nn.Sequential(
            #                 nn.Linear(state_dim, 64),
            #                 nn.Tanh(),
            #                 nn.Linear(64, 64),
            #                 nn.Tanh(),
            #                 nn.Linear(64, action_dim),
            #                 nn.Tanh()
            #             )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )
        # critic
        self.critic = Critic(state_dim)
        # self.critic = nn.Sequential(
        #                 nn.Linear(state_dim, 64),
        #                 nn.Tanh(),
        #                 nn.Linear(64, 64),
        #                 nn.Tanh(),
        #                 nn.Linear(64, 1)
        #             )

    # def set_action_std(self, new_action_std):
    #     if self.has_continuous_action_space:
    #         self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
    #     else:
    #         print("--------------------------------------------------------------------------------------------")
    #         print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
    #         print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    def act(self, state1,state2):
        if self.has_continuous_action_space:
            action_mean, action_sigma = self.actor(state1,state2)
            action_var = action_sigma ** 2
            # action_var = torch.full((self.action_dim,), 0.6).to(device)
            action_var = action_var.repeat(1,2).to(device)
            cov_mat = torch.diag_embed(action_var).to(device)
            # cov_mat = torch.diag(action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state1,state2)
            dist = Categorical(action_probs)

        action = dist.sample()
        action = action.clamp(-1, 1)
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()

    def act_best(self, state1, state2):
        if self.has_continuous_action_space:
            action_mean, action_sigma = self.actor(state1,state2)
            action = action_mean
            action_var = action_sigma ** 2
            # action_var = torch.full((self.action_dim,), 0.6).to(device)
            action_var = action_var.repeat(1,2).to(device)
            cov_mat = torch.diag_embed(action_var).to(device)
            # cov_mat = torch.diag(action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state1,state2)
            action = torch.argmax(action_probs)
            dist = Categorical(action_probs)

        # action = dist.sample()
        action = action.clamp(-1, 1)
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state1, state2, action):

        if self.has_continuous_action_space:
            action_mean, action_sigma = self.actor(state1,state2)
            action_var = action_sigma ** 2
            # action_sigma = torch.full((self.action_dim,), 0.6).to(device)
            action_var = action_var.repeat(1,2).to(device)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state1,state2)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state1,state2)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    # def set_action_std(self, new_action_std):
    #     if self.has_continuous_action_space:
    #         self.action_std = new_action_std
    #         self.policy.set_action_std(new_action_std)
    #         self.policy_old.set_action_std(new_action_std)
    #     else:
    #         print("--------------------------------------------------------------------------------------------")
    #         print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
    #         print("--------------------------------------------------------------------------------------------")

    # def decay_action_std(self, action_std_decay_rate, min_action_std):
    #     print("--------------------------------------------------------------------------------------------")
    #     if self.has_continuous_action_space:
    #         self.action_std = self.action_std - action_std_decay_rate
    #         self.action_std = round(self.action_std, 4)
    #         if (self.action_std <= min_action_std):
    #             self.action_std = min_action_std
    #             print("setting actor output action_std to min_action_std : ", self.action_std)
    #         else:
    #             print("setting actor output action_std to : ", self.action_std)
    #         self.set_action_std(self.action_std)

        # else:
        #     print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        # print("--------------------------------------------------------------------------------------------")

    def select_action(self, state1, state2):

        state1 = torch.FloatTensor(state1).to(device)
        state2 = torch.FloatTensor(state2).to(device)

        if self.has_continuous_action_space:
            with torch.no_grad():
                
                action, action_logprob = self.policy_old.act(state1,state2)

            self.buffer.states.append([state1,state2])
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                # state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state1,state2)
            
            self.buffer.states.append([state1,state2])
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item()

    def select_best_action(self, state1, state2):

        state1 = torch.FloatTensor(state1).to(device)
        state2 = torch.FloatTensor(state2).to(device)

        if self.has_continuous_action_space:
            with torch.no_grad():
                # state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act_best(state1, state2)

            self.buffer.states.append([state1, state2])
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                # state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state1, state2)
            
            self.buffer.states.append([state1, state2])
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
        print("PPO saved")
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
       


