"""
Created by Adex Kwak October 29th 2018
This program is an original. It is not based on a different code.
This game follows the ordinary rules.
제작자: 곽재우 2018/10/29
이 프로그램은 오리지널로, 다른 코드를 기반으로 삼지 않은 것입니다.
이 게임은 일반적인 룰을 따릅니다.
The second player will always win. Therefore, this game is a good experiment to see if the reinforcement learning was proper.
이 게임에서는 후수가 항상 이기므로, 강화학습이 잘 되었는지 보기 위한 좋은 실험군이 됩니다.
"""


import numpy as np

#left = 0, right = 1

class chop:
    def __init__(self):
        
        self.po = np.ones((2,2))
        
        #Allows Forced Quit
        #강제종료를 가능하게 함
        self.force = 0
        
    #Transfer points to one hand from another
    #점수를 두 손 사이에서 전달
    def combine(self, turn, left):
        
        self.po[turn] = np.array([left, np.sum(self.po[turn]) - left])
        
        #If a hand goes over 5, become 0
        #한손이 5를 넘으면, 0이 되겠끔 함
        self.po[self.po >= 5] = 0
        
    #Tap on the opponent's hand
    #상대방의 손을 공격
    def attack(self, turn, FromHand, ToHand):
        
        self.po[1-turn][ToHand] += self.po[turn][FromHand]
        
        self.po[self.po >= 5] = 0
            

################################################################################################################################
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class Chopstick():
 
    def __init__(self): #변하지 않는 변수 같은 것은 이곳에 지정
        self.action_space = spaces.Discrete(9)


        self.seed()
        self.state = None

        self.chop = chop()
        
        self.steps_beyond_done = None
        
        self.turn = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    # 루프 게임을 수행한다
    def step(self, action, rewardsmall):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        #I_r, I_l, O_r, O_l = state
        #각각의 액션의 0~8까지의 행동 지정 
        if action == 0:
            self.chop.combine(self.turn, 0)
        elif action == 1:
            self.chop.combine(self.turn, 1)
        elif action == 2:
            self.chop.combine(self.turn, 2)
        elif action == 3:
            self.chop.combine(self.turn, 3)
        elif action == 4:
            self.chop.combine(self.turn, 4)
        elif action == 5:
            self.chop.attack(self.turn, 0, 1)
        elif action == 6:
            self.chop.attack(self.turn, 0, 0)
        elif action == 7:
            self.chop.attack(self.turn, 1, 1)
        else:
            self.chop.attack(self.turn, 1, 0)
      
        done =  np.sum(self.chop.po[0]) == 0 \
                or np.sum(self.chop.po[1]) == 0 \
            
        self.state = self.chop.po

            
        done = bool(done)
        
        if not done:
            reward = 0
        else:
            if not np.sum(self.chop.po[1 - self.turn]):
                reward = 100
            else:
                reward = -100

        return np.array(self.state), reward + rewardsmall, done, {}

    #게임을 리셋
    def reset(self):
        self.state = np.ones((2,2))
        self.chop.po = np.ones((2,2))
        return np.array(self.state)


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


            

#############################################################################################################################


#DQN이다

import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

EPISODES = 6000


# 카트폴 예제에서의 DQN 에이전트
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False

        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # DQN 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 2000

        # 리플레이 메모리, 최대 크기 4000
        self.memory = deque(maxlen=2000)

        # 모델과 타깃 모델 생성
        self.model = self.build_model()
        self.target_model = self.build_model()

        # 타깃 모델 초기화
        self.update_target_model()

        if self.load_model:
            self.model.load_weights("./save_model/chopstick_dqn.h5")

    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # 타깃 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_size)) #이 부분은 우리의 state_size가 1차원을 넘으면 states = np.zeros((self.batch_size, self.state_size[0], self.state_size[1]......))
        next_states = np.zeros((self.batch_size, self.state_size)) #이 부분도 위와 동일
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        # 현재 상태에 대한 모델의 큐함수
        # 다음 상태에 대한 타깃 모델의 큐함수
        target = self.model.predict(states)
        target_val = self.target_model.predict(next_states)

        # 벨만 최적 방정식을 이용한 업데이트 타깃
        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        self.model.fit(states, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)



if True:
    #  환경 지정, 최대 타임스텝 수가 몇갠지는 위에 episode로
    env =Chopstick()
    state_size = 4
    action_size = 9


    # DQN 에이전트 생성
    agent = [DQNAgent(state_size, action_size), DQNAgent(state_size, action_size)]

    victories, episodes = [], []

    total_victory = [0]
    probability = [0]
    
    turn_env = 1
    for e in range(EPISODES):

        done = False
        # env 초기화
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        env.turn = 0
        
        actionprev = None
        stateprev = None
        rewardprev = None
        next_stateprev = None
        

        while not done:
            if agent[env.turn].render:
                env.render()


                
            # 현재 상태로 행동을 선택 (액션 개수가 매번 다르므로 범위를 초과할 때 재실행)
            rewardsmall = 0
            while True:
                action = agent[env.turn].get_action(state)
                if action <= 4:
                    if action < np.sum(env.chop.po[env.turn]) - 4 or action > np.sum(env.chop.po[env.turn]) or action == env.chop.po[env.turn][1] or action == env.chop.po[env.turn][0]:
                        rewardsmall -= 1
                        continue
                else:
                    if env.chop.po[env.turn][action // 7] <= 0 or env.chop.po[1 - env.turn][action % 2] <= 0:
                        rewardsmall -= 1
                        continue
                
                break
            
            # 선택한 행동으로 환경에서 한 타임스텝 진행
            next_state, reward, done, info = env.step(action, rewardsmall)
            next_state = np.reshape(next_state, [1, state_size])
            # 에피소드가 중간에 끝나면 -100 보상

            # 리플레이 메모리에 샘플 <s, a, r, s'> 저장
            agent[env.turn].append_sample(state, action, reward, next_state, done)
            # 매 타임스텝마다 학습
            if len(agent[env.turn].memory) >= agent[env.turn].train_start:
                agent[env.turn].train_model()

            
            state = next_state

            if done:
                # 각 에피소드마다 타깃 모델을 모델의 가중치로 업데이트
                agent[env.turn].update_target_model()
                
                # 이기면 victory가 1 올라가고 아니면 그대로 유지
                if not np.sum(env.chop.po[1]):
                    victory = 1
                else :
                    victory = 0
                    
                   
                    
                # 에피소드마다 학습 결과 출력
                total_victory.append(total_victory[-1] + victory)
                if e <= 99:
                    probability.append((probability[-1] * (len(total_victory) - 2) + victory) / (len(total_victory) - 1))
                else:
                    probability.append((probability[-1] * 100 - victories[-100] + victory) / 100)
                victories.append(victory)
                episodes.append(e)
                pylab.plot(episodes, probability[1:] , 'b')
                #You need to make a folder in your user folder with the name "save_graph"
                #사용자 파일에 save_graph라는 폴더를 만들어야 합니다.
                pylab.savefig("./save_graph/chopsticks_1.png")
                print("episode:", e, "승리한 플레이어:", victory+1 , "total_victory:" , total_victory[-1] , "  memory length:", len(agent[env.turn].memory), "  epsilon:", str(agent[env.turn].epsilon)[:5], "플레이어 1의 승률", probability[-1] )

                
            env.turn = 1 - env.turn
