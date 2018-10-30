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
class board(object):
    BLACK = 1
    WHITE = 0
    def __init__(self, height, width):
        """
            Constructs a board
            보드 생성
        """
        # Set the height and width of the game board
        #게임판의 가로, 세로 길이를 설정
        self.width = width
        self.height = height
        
        # Creates the list which will contain the pieces each player posesses
        #The order is as followed: White Men, Black Men, White Kings, Black Kings
        #각 플레이어가 가진 말들의 정보를 지닌 리스트 생성
        #순서대로 흰색 말, 검은색 말, 흰색 왕, 검은색 왕
        self.piecelist = [[], [], [], []]
        
        # Set default piece positions
        #처음 말들의 위치를 설정
        for i in range(width):
            self.piecelist[1].append((i, (i+1)%2))
            if i % 2 == 1:
                self.piecelist[1].append((i, 2))
            else:
                self.piecelist[0].append((i, height - 3))
            self.piecelist[0].append((i, height - (i%2) - 1))
            
        # boardState contains the current state of the board for printing/eval
        #현재 보드 출력할 상태에 대한 정보를 지님
        self.boardState = [[' '] * self.width for x in range(self.height)]
        
        #Allows Forced Quit
        #강제종료를 가능하게 함
        self.force = 0
        
        #needed in generating moves
        #행동가능성 생성에 관여
        self.block = 0
        
    
    
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
                        data = [piece, jump]
                        
                        #Check if there is any more that one can jump over
                        #더 뛰어넘을 수는 없는지 확인
                        block2 = 0
                        for move in moves:
                            target2 = (jumpx + move[0], jumpy + move[1])
                            jump2 = (jumpx + 2 * move[0], jumpy + 2 * move[1])
                            
                            #Opponent's piece nearby?    상대방의 말이 가까이 있는가?
                            if target2 in self.piecelist[3 - color] or target2 in self.piecelist[color + (-1) ** color]:
                                #Is there space behind?   그 뒤에 공간이 있는가?
                                if jump2 not in self.piecelist[0] and jump2 not in self.piecelist[1] and jump2 not in self.piecelist[2]and jump2 not in self.piecelist[3] and jump2 != data[-2]:
                                    #Does it not go out of bounds?   보드를 벗어나지 않는가?
                                    if jump2[0] >= 0 and jump2[0] < self.width and jump2[1] >= 0 and jump2[1] < self.height:
                                        block2 = 1
                                        for i in self.jumper(data, moves, color, move):
                                            yield i
                        #There is no more to jump over
                        #더 뛰어넘을 수 있는 것이 없음
                        if block2 == 0:
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

                
            
    def jumper(self, data, moves, color, move):
        """
        Makes the moves of jumping over opponent's piece
        한 번 뛰어넘은 후의 움직임을 생성
        """
        
        data.append((data[-1][0] + 2 * move[0], data[-1][1] + 2 * move[1]))
        
        #Check if there is any more that one can jump over
        #더 뛰어넘을 수는 없는지 확인
        block2 = 0
        for move2 in moves:
            target = (data[-1][0] + move2[0], data[-1][1] + move2[1])
            jump = (data[-1][0] + 2 * move2[0], data[-1][1] + 2 * move2[1])
                            
            #Opponent's piece nearby?    상대방의 말이 가까이 있는가?
            if target in self.piecelist[3 - color] or target in self.piecelist[color + (-1) ** color]:
                #Is there space behind?   그 뒤에 공간이 있는가?
                if jump not in self.piecelist[0] and jump not in self.piecelist[1] and jump not in self.piecelist[2]and jump not in self.piecelist[3] and jump != data[-2]:
                    #Does it not go out of bounds?   보드를 벗어나지 않는가?
                    if jump[0] >= 0 and jump[0] < self.width and jump[1] >= 0 and jump[1] < self.height:
                        block2 = 1
                        for i in self.jumper(data, moves, color, move2):
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
        
        #The final location of the piece
        #말의 최종위치
        self.piecelist[color][self.piecelist[color].index(move[0])] = move[-1]
        
        #Did the piece capture anything?
        #말이 무언가를 포획하였는가?
        if move[0][0] - move[1][0] != 1 and move[0][0] - move[1][0] != -1:
            #Delete the pieces it captured
            #포획한 말들 제거
            for i in range(len(move) - 1):
                if (((move[i][0] + move[i+1][0])/2, (move[i][1] + move[i+1][1])/2)) in self.piecelist[3-color]:
                    self.piecelist[3-color].remove(((move[i][0] + move[i+1][0])/2, (move[i][1] + move[i+1][1])/2))
                elif (((move[i][0] + move[i+1][0])/2, (move[i][1] + move[i+1][1])/2)) in self.piecelist[color + (-1) ** color]:
                    self.piecelist[color + (-1) ** color].remove(((move[i][0] + move[i+1][0])/2, (move[i][1] + move[i+1][1])/2))
                else:
                    raise Exception
        self.updateBoard()
        
        
        
    def promote(self, piece, color):
        '''
        Promotes men to kings if they reached the end
        일반 말들이 끝까지 갔을 시에 왕으로 승격
        '''
        
        if piece[1] == self.width - (1-color) * self.width - 1:
            self.piecelist[color].remove(piece)
            self.piecelist[color+2].append(piece)
            
        

    def updateBoard(self):
        """
            Updates the array containing the board to reflect the current state of the pieces on the board
            현재 보드 상태에 대해 업데이트
        """
        for i in range(self.width):
            for j in range(self.height):
                self.boardState[i][j] = " "
        for piece in self.piecelist[1]:
            self.boardState[piece[1]][piece[0]] = u'b'
        for piece in self.piecelist[0]:
            self.boardState[piece[1]][piece[0]] = u'w'
        for piece in self.piecelist[2]:
            self.boardState[piece[1]][piece[0]] = u'W'
        for piece in self.piecelist[3]:
            self.boardState[piece[1]][piece[0]] = u'B'



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
