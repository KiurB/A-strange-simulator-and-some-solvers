# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib as mpl

def compare_rewards_plot(figsize,dpi,r_random,r_dqn,r_sac):
    
    SMALL_SIZE = 4
    MEDIUM_SIZE = 5
    BIGGER_SIZE = 6

    mpl.rc('font', size=SMALL_SIZE)          
    mpl.rc('axes', titlesize=SMALL_SIZE)    
    mpl.rc('axes', labelsize=MEDIUM_SIZE)   
    mpl.rc('xtick', labelsize=SMALL_SIZE)   
    mpl.rc('ytick', labelsize=SMALL_SIZE)    
    mpl.rc('legend', fontsize=SMALL_SIZE)    
    mpl.rc('figure', titlesize=BIGGER_SIZE)  
    
    fig = plt.figure(figsize=figsize,dpi=dpi)
    plt.plot(r_random,linewidth=0.5,label="random",color="red")
    plt.plot(r_dqn,linewidth=0.5,label="dqn",color="blue")
    plt.plot(r_sac,linewidth=0.5,label="sac",color="orange")
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.ylim((-2000,0))
    plt.legend()
    plt.title("Rewards over time")
    