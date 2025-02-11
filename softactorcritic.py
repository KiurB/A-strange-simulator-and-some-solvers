"""
Created on Sun Nov 17 14:57:02 2024

@author: cure
"""
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import simple_sim as sim
from itertools import count
from memory import Transition, ReplayMemory
import gc

#First actor network

class Critic(nn.Module):
    
    def __init__(self, n_obs, n_act, n_hidden):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(n_obs+n_act, n_hidden*2)
        torch.nn.init.orthogonal_(self.linear1.weight)
        
        self.linear2 = nn.Linear(n_hidden*2, n_hidden)        
        torch.nn.init.orthogonal_(self.linear2.weight)
        
        self.linear3 = nn.Linear(n_hidden, n_hidden)        
        torch.nn.init.orthogonal_(self.linear3.weight)
        
        self.linear4 = nn.Linear(n_hidden, 1)        
        torch.nn.init.orthogonal_(self.linear4.weight)        
        
        self.double()
        
    def forward(self,x):
        x = F.silu(self.linear1(x))
        x = F.silu(self.linear2(x))
        x = F.silu(self.linear3(x))
        return self.linear4(x)
    
class Actor(nn.Module):
    
    def __init__(self, n_obs, n_act, n_hidden):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(n_obs,n_hidden*2)
        torch.nn.init.orthogonal_(self.linear1.weight)
        
        self.linear2 = nn.Linear(n_hidden*2,n_hidden)
        torch.nn.init.orthogonal_(self.linear2.weight)
        
        self.linear3 = nn.Linear(n_hidden,n_hidden)
        torch.nn.init.orthogonal_(self.linear3.weight)
          
        self.linear_mu = nn.Linear(n_hidden, n_act)
        torch.nn.init.orthogonal_(self.linear_mu.weight)
          
        self.linear_sig = nn.Linear(n_hidden, n_act)
        torch.nn.init.orthogonal_(self.linear_sig.weight)
        
        self.min = -10
        self.max = +10

        self.double()
        
    def forward(self,x):
        x = F.silu(self.linear1(x))
        x = F.silu(self.linear2(x))
        x = F.silu(self.linear3(x))
        mu = self.linear_mu(x)
        log_sigma = self.linear_sig(x)
        log_sigma.clamp_(min = self.min, max=self.max)
        
        return mu,log_sigma


class TrainingSAC():

    def __init__(self,
                 BATCH_SIZE = 1024,
                 GAMMA = 0.80,
                 TAU = 0.01,
                 LR = 3e-5,
                 LR_A = 3e-2,
                 n_act = 5,
                 n_obs = 13,
                 n_hidden = 256,
                 act_limits = torch.tensor([[-1000,-1000,-60,-70,-60],
                                            [1000, 1000, 0, 0, -30]])
                 ): 
        
        #key hyper parameters
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.TAU = TAU
        self.LR = LR
        self.LR_A = LR_A
        self.n_act = n_act 
        self.n_obs = n_obs

        #device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "cpu")
        
        #nets and temperature
        self.policy_net = Actor(n_obs,n_act,n_hidden).to(self.device)
        self.critic_net1 = Critic(n_obs,n_act,n_hidden).to(self.device)
        self.critic_net2 = Critic(n_obs,n_act,n_hidden).to(self.device)
        self.target_net1 = Critic(n_obs,n_act,n_hidden).to(self.device)
        self.target_net2 = Critic(n_obs,n_act,n_hidden).to(self.device)
        
        self.log_alpha = torch.tensor(-2.).to(self.device,
                                          dtype=torch.float64)
        self.log_alpha = torch.nn.Parameter(self.log_alpha,requires_grad=True)
        self.alpha = torch.exp(self.log_alpha)
        
        #set target parameters
        self.target_net1.load_state_dict(self.critic_net1.state_dict())
        self.target_net2.load_state_dict(self.critic_net2.state_dict())
        
        #optimizer
        self.optim_policy = optim.AdamW(self.policy_net.parameters(),
                                     lr=self.LR,
                                     amsgrad=True)
        self.optim_Q1 = optim.AdamW(self.critic_net1.parameters(),
                                     lr=self.LR,
                                     amsgrad=True)
        self.optim_Q2 = optim.AdamW(self.critic_net2.parameters(),
                                     lr=self.LR,
                                     amsgrad=True)
        self.optim_alpha = optim.AdamW([self.log_alpha],
                                       lr=self.LR_A,
                                       amsgrad=True)
        self.memory = ReplayMemory(10000)
        
        #keep track
        self.act_limits = act_limits.to(self.device)
        self.state_mu = None
        self.state_std = None
    
    
    def select_action(self, state):
        mu, log_sigma = self.policy_net(state)
        sigma = torch.exp(log_sigma)
        probs = torch.distributions.MultivariateNormal(mu,
                                        scale_tril=torch.diag_embed(sigma,
                                                                    offset=0,
                                                                    dim1=-2,
                                                                    dim2=-1))
        
        actd = probs.rsample().to(self.device)
        act = F.tanh(actd)
        
        log_p = probs.log_prob(actd)
        act_copy = torch.log((1-F.tanh(actd)**2)).sum(axis=1)
        log_p -= act_copy
        
        return act, log_p
    
    
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
        state_action_batch = torch.cat((state_batch,action_batch),dim=1)
        

        #compute targets
        next_state_target1 = torch.zeros((self.BATCH_SIZE,1),
                                          device=self.device,
                                          dtype=torch.float64)
        next_state_target2 = torch.zeros((self.BATCH_SIZE,1),
                                          device=self.device,
                                          dtype=torch.float64)
        nlp_batch = torch.zeros((self.BATCH_SIZE,1),
                                device=self.device,
                                dtype=torch.float64)
        
        with torch.no_grad():
            
            naction_batch, nlp = self.select_action(non_final_next_states)
            nstate_action_batch = torch.cat((non_final_next_states,
                                             naction_batch),
                                            dim=1)
            
            next_state_target1[non_final_mask] =\
                self.target_net1(nstate_action_batch)
            next_state_target2[non_final_mask] =\
                self.target_net2(nstate_action_batch)
            train_q = torch.min(next_state_target1,next_state_target2)
            nlp_batch[non_final_mask] = nlp.unsqueeze(1)
            critic_target = reward_batch + self.GAMMA*(train_q-self.alpha*nlp_batch)
        
        #update critics
        soft_q_1 = self.critic_net1(state_action_batch)
        soft_q_2 = self.critic_net2(state_action_batch)
        loss_critic1 = F.mse_loss(soft_q_1,critic_target)
        loss_critic2 = F.mse_loss(soft_q_2,critic_target)    
        
        #optimize Q-values
        self.optim_Q1.zero_grad()
        self.optim_Q2.zero_grad()
        loss_critic1.backward()
        loss_critic2.backward()
        self.optim_Q1.step()
        self.optim_Q2.step()
        
        #update actor
        repaction, replp = self.select_action(state_batch)
        repstate_action_batch = torch.cat((state_batch,
                                              repaction),
                                              dim=1)
        replp_batch = replp.unsqueeze(1)
        soft_qr_1 = self.critic_net1(repstate_action_batch)
        soft_qr_2 = self.critic_net2(repstate_action_batch)
        min_qr = torch.min(soft_qr_1,soft_qr_2)
        
        #losses
        loss_actor = -min_qr+self.alpha*replp_batch
        loss_actor = torch.mean(loss_actor)
        
        #optimize actor
        self.optim_policy.zero_grad()
        loss_actor.backward()
        self.optim_policy.step()
        
        #loss alpha
        loss_alpha = -self.log_alpha*(replp_batch.detach()-self.n_act)
        loss_alpha = torch.mean(loss_alpha)
        
        #optimize alpha
        self.optim_alpha.zero_grad()
        loss_alpha.backward()
        self.optim_alpha.step()
        
        self.alpha = torch.exp(self.log_alpha).detach().clone()
        
        
    def training_loop(self, n_episodes, k):
        
        simulator = sim.Simulator()
        state_mu = torch.full((1,13),1e-4,device=self.device)
        state_sigma = torch.full((1,13),1e-4,device=self.device)
        c=1
        lr_forget = 0.005
        
        for i in range(n_episodes):
            
            simulator.s_0()                        
            mean_r = 0
            
            #update parameters
            forget = max(1/c,lr_forget)
            old_mu = state_mu
            state_mu = (1-forget)*state_mu+(simulator.state-state_mu)*forget
            state_sigma = (1-forget)*state_sigma+(simulator.state-state_mu)*\
                (simulator.state-state_mu)*forget
            state_std = torch.sqrt(state_sigma)
            
            for t in count():
                
                #prepare soft update
                target_net_state_dict_old1 = self.target_net1.state_dict()
                target_net_state_dict_old2 = self.target_net2.state_dict()
                
                #normalize environment
                norm_state = ((simulator.state-state_mu)/state_std)
                
                #build transition and push
                action, lp = self.select_action(norm_state)
                act = self.act_limits[0]+((action+1)/2)*(self.act_limits[1]-self.act_limits[0])
                simulator.update(act.squeeze())
                
                #update parameters
                if not torch.isnan(simulator.state).any():
                    c += 1
                    forget = max(1/c,lr_forget)
                    old_mu = state_mu
                    state_mu = (1-forget)*state_mu+(simulator.state-state_mu)*forget
                    state_sigma = (1-forget)*state_sigma+(simulator.state-state_mu)*\
                                           (simulator.state-state_mu)*forget
                    state_std = torch.sqrt(state_sigma)
                
                state = norm_state.detach().clone()
                next_state = ((simulator.state-state_mu)/state_std)\
                    .detach().clone()
                reward = simulator.r.unsqueeze(0).detach().clone()
                action_buffer = action.detach().clone()
                lp = lp.detach().clone()
                self.memory.push(state,
                                 action_buffer,
                                 next_state,
                                 reward,
                                 lp
                                 )
                
                #optimize
                skip = 10
                if t%skip==0:
                    self.optimize_model()
                
                    #soft update the weights

                    target_net_state_dict1 = self.critic_net1.state_dict()
                    target_net_state_dict2 = self.critic_net2.state_dict()
                
                    for key in target_net_state_dict1:
                        target_net_state_dict1[key] = target_net_state_dict_old1[key].clone()*\
                            self.TAU+target_net_state_dict_old1[key].clone()*(1-self.TAU)
                    for key in target_net_state_dict2:
                        target_net_state_dict2[key] = target_net_state_dict_old2[key].clone()*\
                            self.TAU+target_net_state_dict_old2[key].clone()*(1-self.TAU)
                
                    self.target_net1.load_state_dict(target_net_state_dict1)
                    self.target_net2.load_state_dict(target_net_state_dict2)
                
                mean_r += reward[0]
                gc.collect()
                
                #end episode
                if torch.isnan(simulator.state).all() or t==k:
                    break
            
            print(mean_r/t)
            print(c)
            print(self.alpha)
            self.state_mu = state_mu
            self.state_std = state_std
    
    
    
    
        
        

