import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    pass

class Critic(nn.Module):
    def __init__(self, lr_critic, input_dims, fc1_dims, fc2_dims, n_agent,action_dim) :
        super(Critic, self).__init__()

        self.fcl = nn.Linear(input_dims + n_agent * action_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, out_features=1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_critic)

    def forward(self, state,action):
        x = torch.cat([state, action], dim=1)
        x = F.relu((self.fcl(x)))
        x = F.relu((self.fc2(x)))
        q = self.q(x)
        return q
    
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkppint_file):
        self.load_state_dict(torch.load(checkppint_file))



class Actor(nn.Module):
    def __init__(self, lr_actor, input_dims, fc1_dims, fc2_dims, action_dim):
        super(Actor, self).__init__()

        self.fcl = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, action_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_actor)

    def forward(self, state):
        x = F.relu((self.fcl(state)))
        x = F.relu((self.fc2(x)))
        mu = torch.softmax(self.pi(x),dim=1)
        return mu
    
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkppint_file):
        self.load_state_dict(torch.load(checkppint_file))
    


class Agent:
    def __init__(self, memo_size, obs_dim, state_dim, n_agent, action_dim,
                 alpha ,beta, fc1_dims,fc2_dims, gamma, tau, batch_size):
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim

        self.actor = Actor(lr_actor=alpha, input_dims=obs_dim, fc1_dims=fc1_dims, 
                           fc2_dims=fc2_dims, action_dim=action_dim)

        self.critic = Critic(lr_critic=beta, input_dims=state_dim, fc1_dims=fc1_dims, 
                             fc2_dims=fc2_dims, n_agent=n_agent, action_dim=action_dim)
        
        self.target_actor = Actor(lr_actor=alpha, input_dims=obs_dim, fc1_dims=fc1_dims, 
                           fc2_dims=fc2_dims, action_dim=action_dim)

        self.target_critic = Critic(lr_critic=beta, input_dims=state_dim, fc1_dims=fc1_dims, 
                             fc2_dims=fc2_dims, n_agent=n_agent, action_dim=action_dim)
    
    def get_action(self, obs):
        single_obs = torch.tensor(data=obs, dtype=torch.float).unsqueeze(0).to(device)
        single_action = self.actor.forward(single_obs)
        noise = torch.randn(self.action_dim).to(device) * 0.2
        single_action = torch.clamp(input=single_action + noise, min=0.0, max=1.0)

        return single_action.detach.cpu().numpy()[0]