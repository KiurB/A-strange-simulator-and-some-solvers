#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 14:57:02 2024

@author: cure
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import namedtuple, deque
import math
import simple_sim as sim
from itertools import count


#set up replay memory

Transition = namedtuple("Transition",
                        ("state","action","next_state","reward"))

class ReplayMemory(object):
    
    def __init__(self,capacity):
        self.memory = deque([],maxlen=capacity)
        
    def push(self,*args):
        self.memory.append(Transition(*args))
        
    def sample(self,batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    

class DQN(nn.Module):
    
    def __init__(self,n_obs,n_actions):
        
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_obs, 256)
        self.layer2 = nn.Linear(256, 1024)
        self.layer4 = nn.Linear(1024,512)
        self.layer3 = nn.Linear(512, n_actions)
        self.double()
        
    def forward(self,x):
        x = F.silu(self.layer1(x))
        x = F.silu(self.layer2(x))
        x = F.silu(self.layer4(x))
        return self.layer3(x)
    

class TrainingDQN():
    
    def __init__(self,BATCH_SIZE=512,GAMMA=0.99,
                     EPS_START=0.80,EPS_END=0.05,EPS_DECAY=1000,
                     TAU=0.05,LR=1e-4,
                     n_actions=243,n_obs=13):
        
        #key hyper parameters
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.TAU = TAU
        self.LR = LR
        self.n_actions = n_actions 
        self.n_obs = n_obs
        
        #device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "cpu")
        
        #nets
        self.policy_net = DQN(n_obs,n_actions).to(self.device)
        self.target_net = DQN(n_obs,n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        #optimizer
        self.optimizer = optim.AdamW(self.policy_net.parameters(),
                                     lr=self.LR,
                                     amsgrad=True)
        self.memory = ReplayMemory(5000)
        
        #keep track
        self.steps_done = 0
        
        #get random policy and simulator
        self.r_policy = sim.Policy(discrete=True)
        self.r_policy.set_random_disc()
        
    def select_action(self,state):
        
        sample = random.random()
        eps_threshold = self.EPS_END+(self.EPS_START-self.EPS_END)*\
            math.exp(-1.*self.steps_done/self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                ind = self.policy_net(state).max(1).indices
                a = self.r_policy.action_space[ind][0].to(self.device)
                return a
        else:
            return self.r_policy.current(self.r_policy).to(self.device)

    
    def optimize_model(self):
        
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: 
                                                not torch.isnan(s).all(),
                                                batch.next_state)),
                                      device=self.device,dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if not torch.isnan(s).all()])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        #get value given action
        state_action_values = self.policy_net(state_batch).gather(1,
                                                                 action_batch)

        #compute value for best action in next state
        next_state_values = torch.zeros(self.BATCH_SIZE,device=self.device,
                                        dtype=torch.float64)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(
                non_final_next_states).max(1).values
        expected_state_action_values = reward_batch +\
                                       self.GAMMA*next_state_values.unsqueeze(1)
        #loss
        loss_f = nn.SmoothL1Loss()
        loss = loss_f(state_action_values,
                      expected_state_action_values)
        
        #optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
                                       
    
    
    def training_loop(self, n_episodes, k):
        
        simulator = sim.Simulator()
        
        for i in range(n_episodes):
            
            simulator.s_0()
            mean_r = 0
            c = 0
            
            for t in count():

                
                #build transition and push
                action = self.select_action(simulator.state.unsqueeze(0))
                simulator.update(action)
                action = ((self.r_policy.action_space==action).all(1))\
                    .nonzero()[0]
                state = simulator.old_state.unsqueeze(0).detach().clone()
                next_state = simulator.state.unsqueeze(0).detach().clone()
                reward = simulator.r.unsqueeze(0).detach().clone()
                self.memory.push(state,
                                 action.unsqueeze(0),
                                 next_state,
                                 reward
                                 )
                
                #optimize
                self.optimize_model()
                
                #soft update the weights
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*\
                        (1-self.TAU)+target_net_state_dict[key]*self.TAU
                self.target_net.load_state_dict(target_net_state_dict)
                
                c += 1
                mean_r += reward[0]
                
                #end episode
                if torch.isnan(simulator.state).all() or t==k:
                    break
                
            print(mean_r/c)
                
            
            
        
        
        
        
        
        
        
        
        
        
