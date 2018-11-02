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
