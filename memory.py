# -*- coding: utf-8 -*-

from collections import namedtuple, deque
import random
#set up replay memory

Transition = namedtuple("Transition",
                        ("state","action","next_state","reward",
                         "logprobability"))

class ReplayMemory(object):
    
    def __init__(self,capacity):
        self.memory = deque([],maxlen=capacity)
        
    def push(self,*args):
        self.memory.append(Transition(*args))
        
    def sample(self,batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)