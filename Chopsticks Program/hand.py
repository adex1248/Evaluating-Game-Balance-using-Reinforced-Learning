import numpy as np

#left = 0, right = 1

class chop:
    def __init__(self):
        self.po = np.ones((2,2))
        self.force = 0
        
    def combine(self, turn, left):
        self.po[turn] = np.array([left, np.sum(self.po[turn]) - left])
        self.po[self.po >= 5] = 0
        
    def attack(self, turn, FromHand, ToHand):
        self.po[1-turn][ToHand] += self.po[turn][FromHand]
        self.po[self.po >= 5] = 0
