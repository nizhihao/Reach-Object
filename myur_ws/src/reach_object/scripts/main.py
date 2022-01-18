#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from env import Ur5
from TD3 import TD3
import numpy as np 
import matplotlib.pyplot as plt
import torch
import time
import argparse
import os


def get_time(start_time):
    m, s = divmod(int(time.time()-start_time), 60)
    h, m = divmod(m, 60)
    print('Total time spent: %d:%02d:%02d' % (h, m, s))


def train(args, env, model):
    if not os.path.exists(args.path_to_model+args.model_name+args.model_date):
        os.makedirs(args.path_to_model+args.model_name+args.model_date)

    #training reward list
    total_reward_list = np.array([])
    #testing reward and steps list
    test_reward_list, test_step_list = np.array([]), np.array([])
    start_time = time.time()
    if args.pre_train:
        #load pre_trained model 
        try:
            model.load_model(args.path_to_model+args.model_name, args.model_date_+'/', args.index)
            print('load model successfully')
        except:
            print('fail to load model, check the path of models')

        print('start random exploration for adding experience')
        state = env.reset()
        for step in range(args.random_exploration):
            state_, action, reward, terminal = env.uniform_exploration(np.random.uniform(-1,1,5)*args.action_bound*5)
            model.store_transition(state,action,reward,state_,terminal)
            state = state_
            if terminal:
                state = env.reset()
        total_reward_list = np.loadtxt(args.path_to_model+args.model_name+args.model_date_+'/reward.txt')
        test_reward_list = np.loadtxt(args.path_to_model+args.model_name+args.model_date_+'/test_reward.txt')
        test_step_list = np.loadtxt(args.path_to_model+args.model_name+args.model_date_+'/test_step.txt')

    print('start training')
    model.mode(mode='train')

    #training for vision observation
    for epoch in range(args.train_epoch):
        state = env.reset()
        total_reward = 0
        for i in range(args.train_step):
            action = model.choose_action(state)
            state_, reward, terminal = env.step(action*args.action_bound)
            model.store_transition(state,action,reward,state_,terminal)
            state = state_
            total_reward += reward
            if model.memory_counter > args.random_exploration:
                model.Learn()
            if terminal:
                state = env.reset()
                
        total_reward_list = np.append(total_reward_list,total_reward)
        print('epoch:', epoch,  '||',  'Reward:', total_reward)

        #begin testing and record the evalation metrics
        if (epoch+1) % args.test_loop == 0:
            model.save_model(args.path_to_model+args.model_name, args.model_date+'/', epoch)
            avg_reward, avg_step = test(args, env, model)
            model.mode(mode='train')
            print('finish testing')
            test_reward_list = np.append(test_reward_list,avg_reward)
            test_step_list = np.append(test_step_list,avg_step)

            np.savetxt(args.path_to_model+args.model_name+args.model_date+'/test_reward.txt',np.array(test_reward_list))
            np.savetxt(args.path_to_model+args.model_name+args.model_date+'/test_step.txt',np.array(test_step_list))
            np.savetxt(args.path_to_model+args.model_name+args.model_date+'/reward.txt',np.array(total_reward_list))
            
            get_time(start_time)    


def test(args, env, model):
    model.mode(mode='test')
    print('start to test the model')
    try:
        model.load_model(args.path_to_model+args.model_name, args.model_date_+'/')
        print('load model successfully')
    except:
        print('fail to load model, check the path of models')

    total_reward_list = []
    steps_list = []
    #testing for vision observation
    for epoch in range(10):
        state = env.reset()
        total_reward = 0
        for step in range(args.test_step):
            action = model.choose_action(state,noise=None)
            state_, reward, terminal = env.step(action*args.action_bound)
            state = state_
            total_reward += reward
            if terminal:
                env.reset()
        total_reward_list.append(total_reward)
        print('testing_epoch:', epoch,  '||',  'Reward:', total_reward)
        
    average_reward = np.mean(np.array(total_reward_list))
    average_step = 0 if steps_list == [] else np.mean(np.array(steps_list))

    return average_reward, average_step


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #Folder name saved as date
    parser.add_argument('--model_name', default='TD3')
    parser.add_argument('--model_date', default='/20220118')
    #Folder stored with trained model weights, which are used for transfer learning
    parser.add_argument('--pre_train', default=True)
    parser.add_argument('--path_to_model', default='/home/user/myur_ws/')
    parser.add_argument('--model_date_', default='/20220113_03')
    parser.add_argument('--index', default=2399, type=int)
    #The maximum action limit
    parser.add_argument('--action_bound', default=np.pi/72, type=float) #pi/36 for reaching
    parser.add_argument('--train_epoch', default=5000, type=int)
    parser.add_argument('--train_step', default=100, type=int)
    parser.add_argument('--test_loop', default=20, type=int)
    parser.add_argument('--test_epoch', default=5, type=int)
    parser.add_argument('--test_step', default=100, type=int)
    #exploration (randome action generation) steps before updating the model
    parser.add_argument('--random_exploration', default=2000, type=int)
    #Wether to use GPU
    parser.add_argument('--cuda', default=True)
    parser.add_argument('--mode', default='train')
    args = parser.parse_args()

    env = Ur5()
    model = TD3(a_dim=env.action_dim,s_dim=env.state_dim,cuda=args.cuda)

    assert args.mode == 'train' or 'test', 'mode: 1.train 2.test'
    if args.mode == 'train': 
        train(args, env, model)

    if args.mode == 'test': 
        env.duration = 0.1
        test(args, env, model)
