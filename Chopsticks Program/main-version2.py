###Main Program###
class gamestart():
    def __init__(self):
        print("Game Start!")
        self.hands = chop()
        print(self.hands.po)
        
        #Corresponds left and right to numbers
        #좌우를 숫자로 대응시킴
        self.handnum = {'left':0, 'right':1}
        self.turn = 0
        #Main Game Loop
        while True:
            #Player's Turn
            #플레이어의 턴
            self.getMove(self.turn)
            print(self.hands.po)
            
            #Force Shutdown
            #강제 종료
            if self.hands.force == 2:
                break
            
            if np.sum(self.hands.po[1]) == 0:
                print('player 1 wins!')
                break
            if np.sum(self.hands.po[0]) == 0:
                print('player 2 wins!')
                break
                
            self.turn = 1 - self.turn

                
                
    def getMove(self, turn):
        '''
        This gets the moves from the player.
        이 함수는 플레이어로부터 행동을 받습니다.
        '''
        print("Player" + str(turn + 1) + "'s turn")
        
        while True:
        
            #Get what type of action the player will do
            #플레이어가 할 행동을 얻음
            a = input("Please input the type of action: 0 for combining and 1 for attacking. Input 2 to quit the game")


            try:
                a = int(a)
            except:
                print('Invalid, try again')
                continue

            #Force the game to quit
            #강제종료
            if a == 2:
                self.hands.force = 2
                return

            
            #Transfering Points
            #점수 전달
            elif a == 0:

                b = input("How much do you want to have in your left hand?")

                try:
                    b = int(b)
                except:
                    print('Invalid, try again')
                    continue

                #See if the left hand's number is reasonable
                #왼쪽 손에 있을 점수가 합리적인지 확인
                if b > np.sum(self.hands.po[turn]) or b > 5 or b < 0:
                    print('Invalid, try again')
                    continue

                #You cannot swap hands or do nothing
                #아무것도 하지 않거나, 손만 서로 바꿀 수는 없음
                elif b == self.hands.po[turn][1] or b == self.hands.po[turn][0]:
                    print("That move is impossible, try again")
                    continue

                else:
                    self.hands.combine(turn, b)
                    return


            #Attack
            #공격
            elif a == 1:

                b = input("Which hand do you want to use to attack?")

                if b != "left" and b != "right":
                    print("Invalid, try again")
                    continue

                c = input("Which hand do you want to attack?")

                if c != "left" and c != "right":
                    print("Invalid, try again")
                    continue

                #Change to numbers
                #숫자로 치환
                b = self.handnum[b]
                c = self.handnum[c]

                if self.hands.po[turn][b] > 0 and self.hands.po[1 - turn][c] > 0:
                    self.hands.attack(turn, b, c)
                    return
                else:
                    print("Invalid, try again")
                    continue
             
             
gamestart = gamestart()
