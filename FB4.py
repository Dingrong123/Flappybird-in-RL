from collections import deque
import torch
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data
from torch.utils.data import sampler
import torch.backends.cudnn as audnn
import torchvision
from torchvision import transforms
import numpy as np
import time
from tensorboardX import SummaryWriter
import torchvision.models as models
from random import randint
import torch.nn as nn # Hyper Parameters:
FRAME_PER_ACTION = 1
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 10000. # timesteps to observe before training
EXPLORE =20000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
INITIAL_EPSILON = 0.01 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH_SIZE = 32 # size of minibatch
UPDATE_TIME = 30
width=80
height=80;
import random


class DeepNetWork(nn.Module):
        def __init__(self):
                super(DeepNetWork, self).__init__() # 需要将事先训练好的词向量载入

                self.conv1 = nn.Sequential( nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8,stride=4,padding=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                )
                torch.nn.init.normal(self.conv1[0].weight, mean=0, std=0.01)

                self.conv2 = nn.Sequential( nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2,padding=1),
                    nn.ReLU(inplace=True),)
                torch.nn.init.normal(self.conv2[0].weight, mean=0, std=0.01)

                self.conv3 = nn.Sequential( nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),)
                torch.nn.init.normal(self.conv3[0].weight, mean=0, std=0.01)
                
                self.fc1=nn.Sequential( nn.Linear(1600,256),
                    nn.ReLU(),)
                torch.nn.init.normal(self.fc1[0].weight, mean=0, std=0.01)
                
                self.out = nn.Linear(256,2)
                torch.nn.init.normal(self.out.weight, mean=0, std=0.01)
                
        def forward(self,x): #btach channel width,weight
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                x = x.view(x.size(0), -1) # 将（batch，outchanel,w,h）展平为（batch，outchanel*w*h）
                x = self.fc1(x)
                out = self.out(x)
                return out

import os

class BrainDQNMain:
    def save(self):
        print("save model param")
        torch.save(self.Q_net.state_dict(), 'paramsnew.pth')
        
    def load(self):
        if os.path.exists("paramsnew.pth"):
            print("load model param")
            self.Q_net.load_state_dict(torch.load('paramsnew.pth'))
            self.Q_netT.load_state_dict(torch.load('paramsnew.pth'))
            
    def __init__(self,actions):
        self.replayMemory = deque() # init some parameters
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = actions  #2？
        self.Q_net=DeepNetWork();
        self.Q_netT=DeepNetWork();
        self.load()
        self.cuda=torch.cuda.is_available()
        self.loss_func=nn.MSELoss()
        self.device=None
        LR=1e-6
        self.optimizer = torch.optim.Adam(self.Q_net.parameters(), lr=LR)
        if self.cuda:
            self.device=torch.device('cuda')
        else:
            slf.device=torch.device('cpu')
        self.Q_net=self.Q_net.to(self.device)
        self.Q_netT=self.Q_netT.to(self.device)
        
        
        
    def train(self): # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replayMemory, BATCH_SIZE)#从deque()中随机取32个样本
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]
        
        # Step 2: calculate y
        y_batch = np.zeros([BATCH_SIZE,1])
        nextState_batch=np.array(nextState_batch)
        nextState_batch=torch.Tensor(nextState_batch)
        action_batch=np.array(action_batch)
        index=action_batch.argmax(axis=1)
        index=np.reshape(index,[BATCH_SIZE,1])
        action_batch_tensor=torch.LongTensor(index)
        QValue_batch = self.Q_netT(nextState_batch.cuda()).cpu()
        QValue_batch=QValue_batch.detach().numpy()
        for i in range(0, BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch[i][0]=reward_batch[i]
            else:
                # 这里的QValue_batch[i]为数组，大小为所有动作集合大小，QValue_batch[i],代表
                # 做所有动作的Q值数组，y计算为如果游戏停止，y=reward[i],如果没停止，则y=reward[i]+gamma*np.max(Qvalue[i])
                # 代表当前y值为当前reward+未来预期最大值*gamma(gamma:经验系数)
                y_batch[i][0]=reward_batch[i] + GAMMA * np.max(QValue_batch[i])
        y_batch=np.array(y_batch)
        y_batch=np.reshape(y_batch,[BATCH_SIZE,1])
        state_batch_tensor=Variable(torch.Tensor(state_batch))
        y_batch_tensor=Variable(torch.Tensor(y_batch))
        y_predict=self.Q_net(state_batch_tensor.cuda()).cpu().gather(1,action_batch_tensor)###根据action得到的当前预测的y值
        
        loss=self.loss_func(y_predict,y_batch_tensor)
        print("LOSS is", loss,y_predict[i],y_batch_tensor[i])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.timeStep % UPDATE_TIME == 0:
                self.Q_netT.load_state_dict(self.Q_net.state_dict())#存储参数
                self.save()

    def setPerception(self,nextObservation,action,reward,terminal):
            #print(nextObservation.shape)
            newState = np.append(self.currentState[1:,:,:],nextObservation,axis = 0)#去掉原来堆叠层的第一张图像，加上最新一张图像
            # newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
            self.replayMemory.append((self.currentState,action,reward,newState,terminal))#在deque中添加这次的信息
            if len(self.replayMemory) > REPLAY_MEMORY:
                self.replayMemory.popleft()#去掉最老的信息
            if self.timeStep > OBSERVE: #经过足够多次的试验后，开始训练
                self.train()
            # print info
            state = ""
            if self.timeStep <= OBSERVE:
                state = "observe"
            elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
                state = "explore"
            else:
                state = "train"
            print ("TIMESTEP", self.timeStep, "/ STATE", state, "/ EPSILON", self.epsilon)
            self.currentState = newState
            self.timeStep += 1

    def getAction(self):
        currentState=torch.Tensor([self.currentState])
        QValue = self.Q_net(currentState.cuda()).cpu()[0] #把黑白图喂进神经网络，得到一个长为2的数组
        action = np.zeros(self.actions) #action为数组 [0,0]
        if self.timeStep % (FRAME_PER_ACTION) == 0:
             if random.random() <= self.epsilon: #episilon-greedy,choose random
                 action_index = random.randrange(self.actions)
                 print("choose random action "+str(action_index))
                 action[action_index] = 1 #选择动作
                 
             else:
                     arr=QValue.detach().numpy()
                     action_index = np.argmax(arr) #episilon-greedy,choose greedy
                     print("choose qnet value action " + str(action_index),'[',arr[0],',',arr[1],']')
                     action[action_index] = 1 #选择动作
                     
        else:
                action[0] = 1  # do nothing
        # change episilon 
        if self.epsilon > FINAL_EPSILON and self.timeStep > (OBSERVE+1):
                self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
          
        return action

    def setInitState(self, observation):
        self.currentState = np.stack((observation, observation, observation, observation), axis=0)#将得到的黑白图堆叠4次？不太明白
        print(self.currentState.shape)


import sys
import cv2
sys.path.append("game/")
import wrapped_flappy_bird as game
import numpy as np


# preprocess raw image to 80*80 gray image
def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return np.reshape(observation,(1,80,80))


def playFlappyBird(): # Step 1: init BrainDQN
    actions = 2
    brain = BrainDQNMain(actions) # Step 2: init Flappy Bird Game
    flappyBird = game.GameState() # Step 3: play game
    
    # Step 3.1: obtain init state
    action0 = np.array([1,0]) # 初始化
    observation0, reward0, terminal = flappyBird.frame_step(action0)#得到这一步的图像，奖励以及是否crush的boolean
    observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)#将观察到的图像变成灰度图
    ret, observation0 = cv2.threshold(observation0,1,255,cv2.THRESH_BINARY)#将灰度图变为完全的黑白图
    brain.setInitState(observation0)#对brain初始化
    print(brain.currentState.shape)

    # Step 3.2: run the game
    while 1!= 0:
        action = brain.getAction()#得到一个数组[0,1]或[1,0]
        nextObservation,reward,terminal = flappyBird.frame_step(action)#得到这一步的图像，奖励以及是否crush的boolean
        nextObservation = preprocess(nextObservation)#预处理一下，和前面一样
        #print(nextObservation.shape)
        brain.setPerception(nextObservation,action,reward,terminal)
        
def main():
    playFlappyBird()

if __name__ == '__main__':
    main()
