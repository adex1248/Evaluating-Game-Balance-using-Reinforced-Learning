            '''
            Created by Adex Kwak October 29th 2018
            This program is based on Carson Wilcox (codeofcarson)'s checkers code.
            For his version of the code, go to:
            https://github.com/codeofcarson/Checkers
            This game follows the rules of English Checkers (a.k.a. American Checkers).
            제작자: 곽재우 2018/10/29
            이 프로그램은 Carson Wilcox (codeofcarson)님의 코드를 참고하여 만들어진 채커 프로그램입니다.
            Carson Wilcox님의 코드를 보시려면, 
            https://github.com/codeofcarson/Checkers
            로 가시면 됩니다.
            이 프로그램은 영국식 채커 (또는 미국식 채커)의 규칙을 따릅니다.
            '''

            #The game board
            #게임판
            class board():
                BLACK = 1
                WHITE = 0
                def __init__(self):
                    """
                        Constructs a board
                        보드 생성
                    """
                    # Set the height and width of the game board
                    #게임판의 가로, 세로 길이를 설정
                    self.width = 8
                    self.height = 8

                    # Creates the list which will contain the pieces each player posesses
                    #The order is as followed: White Men, Black Men, White Kings, Black Kings
                    #각 플레이어가 가진 말들의 정보를 지닌 리스트 생성
                    #순서대로 흰색 말, 검은색 말, 흰색 왕, 검은색 왕
                    self.piecelist = [[], [], [], []]

                    # Set default piece positions
                    #처음 말들의 위치를 설정
                    for i in range(self.width):
                        self.piecelist[1].append((i, (i+1)%2))
                        if i % 2 == 1:
                            self.piecelist[1].append((i, 2))
                        else:
                            self.piecelist[0].append((i, self.height - 3))
                        self.piecelist[0].append((i, self.height - (i%2) - 1))


                    # boardState contains the current state of the board for printing/eval
                    #현재 보드 출력할 상태에 대한 정보를 지님
                    self.boardState = [[' '] * self.width for x in range(self.height)]
                    self.updateBoard()

                    #Allows Forced Quit
                    #강제종료를 가능하게 함
                    self.force = 0

                    #needed in generating moves
                    #행동가능성 생성에 관여
                    self.block = 0

                    self.draw = 0



                def GenMove(self, moves, color):
                    '''
                    Generates actual possible moves of pieces
                    실제로 할 수 있는 행동들 생성

                    Let's look at what the equations mean.
                    아래 수식들이 의미하는 바를 살펴보자.

                    3 - color         color + (-1) ** color

                    0  ->  3              0  ->  1
                    1  ->  2              1  ->  0
                    2  ->  1              2  ->  3
                    3  ->  0              3  ->  2

                    Therefore, these two equations gives out the numbers corresponding to the list with pieces of the opponent.
                    따라서, 이 두 수식은 상대방의 말들의 정보가 담긴 리스트에 해당하는 숫자들을 불러옴을 알 수 있다.
                    '''


                    for piece in self.piecelist[color]:

                        #First, we check if there is any piece we can capture. If there is, we must capture at least one piece and there cannot
                        #simply move.
                        #우선적으로, 포획할 수 있는 말이 있는지 살핍니다. 있으면, 무조건 잡아야 하므로 단순히 움직이기만 할 수는 없습니다.
                        for move in moves:
                            targetx = piece[0] + move[0]
                            targety = piece[1] + move[1]
                            target = (targetx, targety)

                            #Figure whether the piece in front is the opponent's
                            #말 바로 앞의 말이 상대방의 말인지 확인
                            if target not in self.piecelist[3 - color] and target not in self.piecelist[color + (-1) ** color]:
                                continue

                            else:
                                jumpx = targetx + move[0]
                                jumpy = targety + move[1]

                                #The piece must go in-bounds
                                #보드 내에서 움직여야 함
                                if jumpx < 0 or jumpx >= self.width or jumpy < 0 or jumpy >= self.height:
                                    continue
                                jump = (jumpx, jumpy)

                                #If there is a piece behind the opponent's, we cannot capture
                                #상대방의 말 뒤에 다른 말이 있으면, 포획은 불가
                                if jump not in self.piecelist[0] and jump not in self.piecelist[1] and jump not in self.piecelist[2]and jump not in self.piecelist[3]:
                                    #blocks any more moves not involving capture, since a capture is possible!
                                    #포획할 수 있으므로 포획을 포함하지 않는 움직임들 생성 억제
                                    self.block = 1    

                                    #list of piece jumped over   잡은 말들에 대한 리스트
                                    stop_back = [ ( (piece[0] + jumpx)//2, (piece[1] + jumpy)//2 ) ]     ###############changed

                                    #Check if there is any more that one can jump over
                                    #더 뛰어넘을 수는 없는지 확인
                                    block2 = 1
                                    for move in moves:
                                        data = [piece, jump]
                                        target2 = (jumpx + move[0], jumpy + move[1])
                                        jump2 = (jumpx + 2 * move[0], jumpy + 2 * move[1])

                                        #Opponent's piece nearby?    상대방의 말이 가까이 있는가?
                                        if target2 in self.piecelist[3 - color] or target2 in self.piecelist[color + (-1) ** color]:
                                            #Is there space behind?   그 뒤에 공간이 있는가?
                                            if jump2 not in self.piecelist[0] and jump2 not in self.piecelist[1] and jump2 not in self.piecelist[2]and jump2 not in self.piecelist[3] and jump2 != data[-2]:
                                                #Does it not go out of bounds?   보드를 벗어나지 않는가?
                                                if jump2[0] >= 0 and jump2[0] < self.width and jump2[1] >= 0 and jump2[1] < self.height:
                                                    #Does the piece not jump over one already taken?   이미 잡은 말을 또 잡지는 않는가?
                                                    if target2 not in stop_back:    #############changed
                                                        block2 = 0
                                                        stop_back.append(target2)
                                                        for i in self.jumper(data, moves, color, move, stop_back):
                                                            yield i
                                    #There is no more to jump over
                                    #더 뛰어넘을 수 있는 것이 없음
                                    if block2:
                                        yield data


                    #If there is no piece to be captured
                    #포획가능 말이 없을 시
                    if self.block != 1:
                        for piece in self.piecelist[color]:
                            for move in moves:
                                targetx = piece[0] + move[0]
                                targety = piece[1] + move[1]
                                target = (targetx, targety)

                                #The piece must go in-bounds
                                #보드 내에서 움직여야 함
                                if target not in self.piecelist[0] and target not in self.piecelist[1] and target not in self.piecelist[2]and target not in self.piecelist[3]:
                                    if targetx >= 0 and targetx < self.width and targety >= 0 and targety < self.height:
                                        yield ([piece, target])



                def jumper(self, data, moves, color, move, stop_back):
                    """
                    Makes the moves of jumping over opponent's piece
                    한 번 뛰어넘은 후의 움직임을 생성
                    """

                    data.append((data[-1][0] + 2 * move[0], data[-1][1] + 2 * move[1]))
                    a = data

                    #Check if there is any more that one can jump over
                    #더 뛰어넘을 수는 없는지 확인
                    block2 = 0
                    for move2 in moves:
                        data = a
                        target = (data[-1][0] + move2[0], data[-1][1] + move2[1])
                        jump = (data[-1][0] + 2 * move2[0], data[-1][1] + 2 * move2[1])

                        #Opponent's piece nearby?    상대방의 말이 가까이 있는가?
                        if target in self.piecelist[3 - color] or target in self.piecelist[color + (-1) ** color]:
                            #Is there space behind?   그 뒤에 공간이 있는가?
                            if jump not in self.piecelist[0] and jump not in self.piecelist[1] and jump not in self.piecelist[2]and jump not in self.piecelist[3] and jump != data[-2]:
                                #Does it not go out of bounds?   보드를 벗어나지 않는가?
                                if jump[0] >= 0 and jump[0] < self.width and jump[1] >= 0 and jump[1] < self.height:
                                    #Does the piece not jump over one already taken?   이미 잡은 말을 또 잡지는 않는가?
                                    if target not in stop_back:    #############changed
                                        block2 = 1
                                        stop_back.append(target)
                                        for i in self.jumper(data, moves, color, move2, stop_back):
                                            yield i

                    #There is no more to jump over
                    #더 뛰어넘을 수 있는 것이 없음
                    if block2 == 0:
                        yield data



                def Obtain(self, moves, color):
                    '''
                    Moves all capable moves into a list
                    가능한 모든 움직임을 리스트에 저장
                    '''
                    available1 = []
                    available2 = []

                    #If there is a piece in the kings list that can capture, the men's movements will still be made.
                    #Therefore, an element directly from the class is needed to remove the men's movements.
                    #왕들 중 포획이 가능한 것이 있다면, 일반 말들의 움직임은 생성될 것이다.
                    #따라서, 클래스에 직접적으로 지정된 원소로 일반 말들의 움직임을 지워야 한다.
                    self.block = 0

                    #Generate moves of men
                    #일반 말들의 움직임 생성
                    for i in self.GenMove(moves, color):
                        available1.append(i)
                    #See if there were any captures by men
                    #일반 말들이 포획했는지 확인
                    temp1 = self.block

                    #Generate moves of kings
                    #왕들의 움직임 생성
                    for i in self.GenMove(((-1, -1), (1, -1), (-1, 1), (1, 1)), color+2):
                        available2.append(i)

                    #See explanation above(234~237)
                    #위 설명 참조(234~237)
                    if temp1 == 0 and self.block == 1:
                        available = available2
                    else:
                        available = available1 + available2

                    return available



                def move(self, move, color):
                    '''
                    Moves the piece selected
                    선택된 말을 움직임
                    '''
                    reward = 0
                    #The final location of the piece
                    #말의 최종위치
                    self.piecelist[color][self.piecelist[color].index(move[0])] = move[-1]

                    #Did the piece capture anything?
                    #말이 무언가를 포획하였는가?
                    if move[0][0] - move[1][0] != 1 and move[0][0] - move[1][0] != -1:
                        #Delete the pieces it captured
                        #포획한 말들 제거

                        for i in range(len(move) - 1):
                            if (((move[i][0] + move[i+1][0])//2, (move[i][1] + move[i+1][1])//2)) in self.piecelist[3-color]:
                                reward += 30 if color <= 1 else 20
                                self.draw = 0
                                self.piecelist[3-color].remove(((move[i][0] + move[i+1][0])/2, (move[i][1] + move[i+1][1])/2))
                            elif (((move[i][0] + move[i+1][0])//2, (move[i][1] + move[i+1][1])//2)) in self.piecelist[color + (-1) ** color]:
                                reward += 20 if color <= 1 else 30
                                self.draw = 0
                                self.piecelist[color + (-1) ** color].remove(((move[i][0] + move[i+1][0])/2, (move[i][1] + move[i+1][1])/2))
                            else:
                                print(move, self.piecelist, color)
                                raise Exception
                    self.updateBoard()
                    return self.promote(move[-1], color, reward) if color <= 1 else reward



                def promote(self, piece, color, reward):
                    '''
                    Promotes men to kings if they reached the end                  
                    일반 말들이 끝까지 갔을 시에 왕으로 승격
                    '''
                    temp = 0 if color == 0 else 7                 #################changed
                    if piece[1] == temp:
                        self.piecelist[color].remove(piece)
                        self.piecelist[color+2].append(piece)
                        self.draw = 0
                        reward += 15
                    return reward    


                def updateBoard(self):
                    """
                        Updates the array containing the board to reflect the current state of the pieces on the board
                        현재 보드 상태에 대해 업데이트
                    """
                    for i in range(self.width):
                        for j in range(self.height):
                            self.boardState[i][j] = -1
                    for piece in self.piecelist[1]:
                        self.boardState[piece[1]][piece[0]] = 1
                    for piece in self.piecelist[0]:
                        self.boardState[piece[1]][piece[0]] = 0
                    for piece in self.piecelist[2]:
                        self.boardState[piece[1]][piece[0]] = 2
                    for piece in self.piecelist[3]:
                        self.boardState[piece[1]][piece[0]] = 3



                def printBoard(self):
                    """
                        Prints the game board
                        보드를 출력
                    """
                    print(self.unicode())



                def unicode(self):
                    """
                        Contains the unicode etc. for printing the board
                        보드 출력에 대한 정보를 지님
                    """
                    # Updates Game board
                    self.updateBoard()
                    lines = []
                    # This prints the numbers at the top of the Game Board
                    lines.append('      ' + '    '.join(map(str, range(self.width))))
                    # Prints the top of the gameboard in unicode
                    lines.append(u'  ╭' + (u'---┬' * (self.width-1)) + u'---╮')

                    # Print the boards rows
                    for num, row in enumerate(self.boardState[:-1]):
                        lines.append(chr(num+65) + u' │ ' + u' │ '.join(row) + u' │')
                        lines.append(u'  ├' + (u'---┼' * (self.width-1)) + u'---┤')

                    #Print the last row
                    lines.append(chr(self.height+64) + u' │ ' + u' │ '.join(self.boardState[-1]) + u' │')

                    # Prints the final line in the board
                    lines.append(u'  ╰' + (u'---┴' * (self.width-1)) + u'---╯')
                    return '\n'.join(lines)
                    
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class Checkers():
    def __init__(self): #변하지 않는 변수 같은 것은 이곳에 지정
        self.action_space = spaces.Discrete(20)                          #########지속적 변경
        self.seed()
        self.state = None

        self.board = board()
        self.board.updateBoard()
        
        self.steps_beyond_done = None
        
        self.turn = 0
        
        self.available = []
        
    def getMove(self, color):
        #white = 0, black = 1
        #백 = 0, 흑 = 1
        direction = 2 * color - 1
        self.available = self.board.Obtain(((-1, direction), (1, direction)), color)
        
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
     # 루프 게임을 수행한다
    def step(self, action, rewardsmall):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        #I_r, I_l, O_r, O_l = state
        #각각의 액션의 0~8까지의 행동 지정
        
        self.board.draw += 1
        
        if self.available[action][0] in self.board.piecelist[self.turn]:
            reward = self.board.move(self.available[action], self.turn)
        else:
            reward = self.board.move(self.available[action], self.turn + 2)
            
        done = (len(self.board.piecelist[3 - self.turn]) == 0 and len(self.board.piecelist[self.turn + (-1) ** self.turn])  == 0) or self.board.draw == 80
        
        self.state = self.board.boardState
        
        done = bool(done)
        
        if done:
            reward += 0 if self.draw == 80 else 1000

                
        return np.array(self.state), reward + rewardsmall, done, {}
    
    #게임을 리셋
    def reset(self,start_piece,start_board):
        # Creates the list which will contain the pieces each player posesses
        #The order is as followed: White Men, Black Men, White Kings, Black Kings
        #각 플레이어가 가진 말들의 정보를 지닌 리스트 생성
        #순서대로 흰색 말, 검은색 말, 흰색 왕, 검은색 왕
        self.board.piecelist = [[], [], [], []]
        
        # Set default piece positions
        #처음 말들의 위치를 설정
        for i in range(self.board.width):
            self.board.piecelist[1].append((i, (i+1)%2))
            if i % 2 == 1:
                self.board.piecelist[1].append((i, 2))
            else:
                self.board.piecelist[0].append((i, self.board.height - 3))
            self.board.piecelist[0].append((i, self.board.height - (i%2) - 1))
            
            
        # boardState contains the current state of the board for printing/eval
        #현재 보드 출력할 상태에 대한 정보를 지님
        self.board.boardState = [[' '] * self.board.width for x in range(self.board.height)]      
        self.board.updateBoard()
        self.state = self.board.boardState
        return self.state
        
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
        self.epsilon_decay = 0.99998
        self.epsilon_min = 0.01
        self.batch_size = 32
        self.train_start = 50000

        # 리플레이 메모리, 최대 크기 400000
        self.memory = deque(maxlen=400000)

        # 모델과 타깃 모델 생성
        self.model = self.build_model()
        self.target_model = self.build_model()

        # 타깃 모델 초기화
        self.update_target_model()


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
    env = Checkers()
    
    start_piece = env.board.piecelist
    start_board = env.board.boardState
    
    EPISODES = 1000000
    
    state_size = 64
    action_size = 20


    # DQN 에이전트 생성
    agent = [DQNAgent(state_size, action_size), DQNAgent(state_size, action_size)]

    victories, episodes = [], []

    probability0 = [0]
    probability1 = [0]
    probability2 = [0]
    
    for e in range(EPISODES):

        done = False
        # env 초기화
        state = env.reset(start_piece,start_board)
        state = np.reshape(state, [1, state_size])
        env.turn = 0
        
        env.draw = 0
        
        actionprev = None
        stateprev = None
        next_stateprev = None

        while not done and env.draw < 80:
            if agent[env.turn].render:
                env.render()
        
            env.getMove(env.turn)
                        
            temp = 0
            if len(env.available) == 0:
                temp = 1
                done = True
                
            if temp == 0:
                
                # 현재 상태로 행동을 선택
                rewardsmall = 0
                while True:
                    action = agent[env.turn].get_action(state)
                    if action >= len(env.available):
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

                actionprev = action
                stateprev = state
                next_stateprev = next_state

            if done or temp:
                # 각 에피소드마다 타깃 모델을 모델의 가중치로 업데이트
                agent[env.turn].update_target_model()
                
                # 이기면 victory가 1 올라가고 아니면 그대로 유지
                victory = 0 if (env.turn) else 1
                if env.board.draw == 80 or temp: victory = 2 
                    
                if victory != 2:
                    agent[1 - env.turn].append_sample(stateprev, actionprev, -1000, next_stateprev, not done)
                    
                    # 에피소드마다 학습 결과 출력
                    if e <= 99:
                        probability0.append( (probability0[-1] * (e) + victory) / (e + 1) )
                        probability1.append( (probability1[-1] * (e) + (1 - victory)) / (e + 1) )
                    else:
                        i = victories[-100] if victories[-100] != 2 else 0
                        j = victories[-100] if victories[-100] != 2 else 1

                        probability0.append((probability0[-1] * 100 - i + victory) / 100)
                        probability1.append((probability1[-1] * 100 - (1 - j) + (1 - victory)) / 100)
                        
                else:
                    if e <= 99:
                        probability0.append(probability0[-1] * e / (e + 1))
                        probability1.append(probability1[-1] * e / (e + 1))
                    else:
                        i = victories[-100] if victories[-100] != 2 else 0
                        j = victories[-100] if victories[-100] != 2 else 1

                        probability0.append((probability0[-1] * 100 - i) / 100)
                        probability1.append((probability1[-1] * 100 - (1 - j)) / 100)
                probability2.append(1 - probability0[-1] - probability1[-1])
                    
                if victory == 2:
                    a = 'draw'
                elif victory == 1:
                    a = '1p'
                else:
                    a = '2p'
                victories.append(victory)
                episodes.append(e)
                pylab.plot(episodes, probability0[1:], 'b')
                pylab.plot(episodes, probability1[1:], 'r')
                pylab.plot(episodes, probability2[1:], 'g')
                pylab.savefig("./save_graph/checkers_1.png")
                print("episode:", e, "승리:", a , "  epsilon:", str(agent[env.turn].epsilon)[:5], "  memory length:", len(agent[env.turn].memory), "p1", probability0[-1], "p2", probability1[-1],"draw", probability2[-1] )

                # "  memory length:", len(agent[env.turn].memory),
            env.turn = 1 - env.turn
    
