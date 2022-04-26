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
        self.pre_actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.pre_actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init=0.6):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128,64)
        self.mu_head = nn.Linear(64, action_dim)
        self.sigma_head = nn.Linear(64, 1)
        self.action_std_init = action_std_init

    def forward(self, s, u_hat):
        x = torch.cat([s, u_hat], -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))

        mu = F.softsign(self.mu_head(x))
        sigma = self.action_std_init*torch.sigmoid(self.sigma_head(x))

        return mu, sigma

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.state_value= nn.Linear(64, 1)

    def forward(self, s, u):
        x = torch.cat([s,u],-1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        value = self.state_value(x)
        return value

class OM(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init=0.6):
        super(OM, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128,64)
        self.mu_head = nn.Linear(64, action_dim)
        self.sigma_head = nn.Linear(64, 1)
        self.action_std_init = action_std_init

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))

        mu = F.softsign(self.mu_head(x))
        sigma = self.action_std_init*torch.sigmoid(self.sigma_head(x))

        return mu, sigma

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        if has_continuous_action_space:
            self.action_dim = action_dim

        # actor
        if has_continuous_action_space :
            self.actor = Actor(state_dim, action_dim, action_std_init)

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
        self.critic = Critic(state_dim,action_dim)

        # om
        self.om = OM(state_dim, action_dim, action_std_init)


    def forward(self):
        raise NotImplementedError
    
    def act(self, state, opponent_state):
        if self.has_continuous_action_space:
            pre_mean, pre_sigma = self.om(opponent_state)
            pre_var = pre_sigma ** 2
            pre_var = pre_var.repeat(1,2).to(device)
            pre_mat = torch.diag_embed(pre_var).to(device)
            pre_dist = MultivariateNormal(pre_mean, pre_mat)
            pre_action = pre_dist.sample()
            pre_action = pre_action.clamp(-1, 1)
            action_mean, action_sigma = self.actor(state,pre_action[0])
            action_var = action_sigma ** 2
            action_var = action_var.repeat(1,2).to(device)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action = action.clamp(-1, 1)
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach(), pre_action.detach()

    def act_best(self, state, opponent_state):
        if self.has_continuous_action_space:
            pre_mean, _ = self.om(opponent_state)
            pre_action = pre_mean
            pre_action = pre_action.clamp(-1, 1)

            action_mean, action_sigma = self.actor(state,pre_action)
            action = action_mean
            action_var = action_sigma ** 2
            action_var = action_var.repeat(1,2).to(device)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            action = torch.argmax(action_probs)
            dist = Categorical(action_probs)

        # action = dist.sample()
        action = action.clamp(-1, 1)
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach(), pre_action.detach()
    
    def evaluate(self, state, opponent_state, action):

        if self.has_continuous_action_space:
            pre_mean, pre_sigma = self.om(opponent_state)
            pre_var = pre_sigma ** 2
            pre_var = pre_var.repeat(1,2).to(device)
            pre_mat = torch.diag_embed(pre_var).to(device)
            pre_dist = MultivariateNormal(pre_mean, pre_mat)
            pre_action = pre_dist.sample()
            pre_action = pre_action.clamp(-1, 1)

            action_mean, action_sigma = self.actor(state,pre_action)
            action_var = action_sigma ** 2
            action_var = action_var.repeat(1,2).to(device)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state,action)
        
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
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic},
                         {'params': self.policy.om.parameters(), 'lr': lr_actor}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state, opponent_state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                opponent_state = torch.FloatTensor(opponent_state).to(device)
                action, action_logprob, pre_action = self.policy_old.act(state, opponent_state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.pre_actions.append(pre_action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state, opponent_state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item()

    def select_best_action(self, state, opponent_state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                opponent_state = torch.FloatTensor(opponent_state).to(device)
                action, action_logprob, pre_action = self.policy_old.act_best(state,opponent_state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.pre_actions.append(pre_action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                opponent_state = torch.FloatTensor(opponent_state).to(device)
                action, action_logprob = self.policy_old.act(state,opponent_state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item()

    def update(self, opponent):
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
        old_opponent_states = torch.squeeze(torch.stack(opponent.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_pre_actions = torch.squeeze(torch.stack(opponent.buffer.pre_actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_opponent_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy + 0.5 * self.MseLoss(old_actions, old_pre_actions)
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

    def clear(self):
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
        print("PPO saved")
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        