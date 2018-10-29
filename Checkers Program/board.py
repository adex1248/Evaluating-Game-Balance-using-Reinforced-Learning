

'''
This game follows the rules of English Checkers.
'''



class board(object):
    BLACK = 1
    WHITE = 0
    def __init__(self, height, width):
        """
            Constructs a board
        """
        # Set the height and width of the game board
        self.width = width
        self.height = height
        # Creates lists which will contain the pieces each player posesses
        self.piecelist = [[], [], [], []]                ##white, black, whiteking, blackking
        # Set default piece positions
        for i in range(width):
            self.piecelist[1].append((i, (i+1)%2))
            if i % 2 == 1:
                self.piecelist[1].append((i, 2))
            else:
                self.piecelist[0].append((i, height - 3))
            self.piecelist[0].append((i, height - (i%2) - 1))
        # boardState contains the current state of the board for printing/eval
        self.boardState = [[' '] * self.width for x in range(self.height)]
        self.force = 0      #allows forced quit
        self.debug = 0
        self.block = 0
    
    def GenMove(self, moves, color):
        '''
        Generates actual possible moves of pieces
        '''
        for piece in self.piecelist[color]:
            for move in moves:
                targetx = piece[0] + move[0]
                targety = piece[1] + move[1]
                target = (targetx, targety)
                    
                if target not in self.piecelist[3 - color] and target not in self.piecelist[color + (-1) ** color]:                 #can only jump over opponent's piece
                    continue
                    
                else:
                    jumpx = targetx + move[0]
                    jumpy = targety + move[1]
                    if jumpx < 0 or jumpx >= self.width or jumpy < 0 or jumpy >= self.height:
                        continue
                    jump = (jumpx, jumpy)
                    if target in self.piecelist[color] or target in self.piecelist[2-color]:
                        continue
                    
                    if jump not in self.piecelist[0] and jump not in self.piecelist[1] and jump not in self.piecelist[2]and jump not in self.piecelist[3]:
                        self.block = 1                     #blocks any more moves not involving capture, since a capture is possible!
                        yield [piece, jump]
                        
                        for i in self.jumper(jump, moves, color, [piece, jump]):
                            yield i
        
        if self.block != 1:
            self.debug += 1
            for piece in self.piecelist[color]:
                for move in moves:
                    targetx = piece[0] + move[0]
                    targety = piece[1] + move[1]
                    target = (targetx, targety)
                    if target not in self.piecelist[0] and target not in self.piecelist[1] and target not in self.piecelist[2]and target not in self.piecelist[3]:
                        if targetx >= 0 and targetx < self.width and targety >= 0 and targety < self.height:
                            yield ([piece, target])

                
            
    def jumper(self, piece, moves, color, data):
        """
        Makes the moves of jumping over opponent's piece
        """
        for move in moves:
            targetx = piece[0] + move[0]
            targety = piece[1] + move[1]
            target = (targetx, targety)
            if target in self.piecelist[color] or target in self.piecelist[2 - color]:                 #can only jump over opponent's piece
                continue
                    

            jumpx = targetx + move[0]
            jumpy = targety + move[1]
            if jumpx < 0 or jumpx >= self.width or jumpy < 0 or jumpy >= self.height:
                continue
            jump = (jumpx, jumpy)
                
            if jump not in self.piecelist[0] and jump not in self.piecelist[1] and jump not in self.piecelist[2] and jump not in self.piecelist[3] and jump != data[-2]:
                data.append(jump)
                yield data
                
                a = []
                b = []
                for move in moves:
                    a.append((jumpx + move[0], jumpy + move[1]))
                    b.append((jumpx + move[0] * 2, jumpy + move[1] * 2))
                        
                for i in range(len(a)):
                    if a[i] in self.piecelist[3 - color] + self.piecelist[color + (-1) ** color] and a[i] != target:
                        if b[i] not in self.piecelist[0] and b[i] not in self.piecelist[1] and b[i] not in self.piecelist[2]and b[i] not in self.piecelist[3] and b[i] not in data:
                            if b[i][0] >= 0 and b[i][0] < self.width and b[i][1] >= 0 and b[i][1] < self.height:
                                for i in self.jumper2(jump, moves, color, data, a[i], b[i]):
                                    yield i

                    
    def jumper2(self, piece, moves, color, data, c, d):
        
        jump = d
        data.append(jump)
        yield data
        a = []
        b = []
        for move in moves:
            a.append((jump[0] + move[0], jump[1] + move[1]))
            b.append((jump[0] + move[0] * 2, jump[1] + move[1] * 2))
                        
        for i in range(len(a)):
            if a[i] in self.piecelist[3 - color] + self.piecelist[color + (-1) ** color] and a[i] != c:
                if b[i] not in self.piecelist[0] and b[i] not in self.piecelist[1] and b[i] not in self.piecelist[2]and b[i] not in self.piecelist[3] and b[i] not in data:
                    if b[i][0] >= 0 and b[i][0] < self.width and b[i][1] >= 0 and b[i][1] < self.height:
                        for i in self.jumper2(jump, moves, color, [piece, jump], a[i], b[i]):
                            yield i
    
    
                    
    def Obtain(self, moves, color):
        '''
        Moves all capable moves into a list
        '''
        available1 = []
        available2 = []
        self.block = 0
        for i in self.GenMove(moves, color):
            available1.append(i)
        temp1 = self.block
        for i in self.GenMove(((-1, -1), (1, -1), (-1, 1), (1, 1)), color+2):
            available2.append(i)
        if temp1 == 0 and self.block == 1:
            available = available2
        else:
            available = available1 + available2
        return available
    
    def move(self, move, color):
        self.piecelist[color][self.piecelist[color].index(move[0])] = move[-1]
        if move[0][0] - move[1][0] != 1 and move[0][0] - move[1][0] != -1:
            for i in range(len(move) - 1):
                if (((move[i][0] + move[i+1][0])/2, (move[i][1] + move[i+1][1])/2)) in self.piecelist[3-color]:
                    self.piecelist[3-color].remove(((move[i][0] + move[i+1][0])/2, (move[i][1] + move[i+1][1])/2))
                elif (((move[i][0] + move[i+1][0])/2, (move[i][1] + move[i+1][1])/2)) in self.piecelist[color + (-1) ** color]:
                    self.piecelist[color + (-1) ** color].remove(((move[i][0] + move[i+1][0])/2, (move[i][1] + move[i+1][1])/2))
                else:
                    raise Exception
        self.updateBoard()
        
    def promote(self, piece, color):
        if piece[1] == self.width - (1-color) * self.width - 1: #(4,7)
            self.piecelist[color].remove(piece)
            self.piecelist[color+2].append(piece)
            
        

    def updateBoard(self):
        """
            Updates the array containing the board to reflect the current state of the pieces on the
            board
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
            Prints the game board to stdout
        """
        print(self.unicode())
        
    def unicode(self):
        """
            Contains the unicode and other BS for printing the board
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
