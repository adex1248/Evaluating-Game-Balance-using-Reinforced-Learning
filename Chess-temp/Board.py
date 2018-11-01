#The game board
#게임판
class board(object):
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
        self.piecelist = [  [ [],[],[],[],[],[] ] , [ [],[],[],[],[],[] ]  ]
        self.type_piece = {0:'p', 1:'b', 2:'h', 3:'r', 4:'q', 5:'k'}
        
        # Set default piece positions
        #처음 말들의 위치를 설정
        for i in range(self.width):
            self.piecelist[0][0].append([i, 6])    #pawn
            self.piecelist[1][0].append([i, 1])
            if i == 0 or i == 7:
                self.piecelist[0][3].append([i, 7])     #rook
                self.piecelist[1][3].append([i, 0])
            elif i == 1 or i == 6:
                self.piecelist[0][2].append([i, 7])     #horse
                self.piecelist[1][2].append([i, 0])
            elif i == 2 or i == 5:
                self.piecelist[0][1].append([i, 7])     #bishop
                self.piecelist[1][1].append([i, 0])
            elif i == 3:
                self.piecelist[0][4].append([i, 7])       #queen
                self.piecelist[1][4].append([i, 7])
            else:
                self.piecelist[0][5].append([i, 7])       #king
                self.piecelist[1][5].append([i, 7])
            
        # boardState contains the current state of the board for printing/eval
        #현재 보드 출력할 상태에 대한 정보를 지님
        self.boardState = [['  '] * self.width for x in range(self.height)]
        
        #Allows Forced Quit
        #강제종료를 가능하게 함
        self.force = 0
        
        #needed in generating moves
        #행동가능성 생성에 관여
        self.block = 0
        
    
    
    #def GenMove(self, moves, color):
    
    
    #def move(self, move, color):
        '''
        Moves the piece selected
        선택된 말을 움직임
        '''
        
            
        

    def updateBoard(self):
        """
            Updates the array containing the board to reflect the current state of the pieces on the board
            현재 보드 상태에 대해 업데이트
        """
        for i in range(self.width):
            for j in range(self.height):
                self.boardState[i][j] = " "
        for pieces in range(len(self.piecelist[0])):
            for piece in self.piecelist[pieces]:
                self.boardState[piece[1]][piece[0]] = 'W' + self.type_piece[pieces]
        for pieces in range(len(self.piecelist[1])):
            for piece in self.piecelist[pieces]:
                self.boardState[piece[1]][piece[0]] = 'B' + self.type_piece[pieces]



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
        lines.append('      ' + '    '.join(map(str, range(1,9))))
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



b = board()
b.printBoard()
