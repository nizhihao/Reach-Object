#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np 
import matplotlib.pyplot as plt
total_reward_list = np.loadtxt('/home/user/myur_ws/TD3/20220113_03/reward.txt')
total_reward_list = total_reward_list

plt.figure()
plt.plot(np.arange(len(total_reward_list)), total_reward_list)
plt.ylabel('Total_reward')
plt.xlabel('training epoch')
plt.show()