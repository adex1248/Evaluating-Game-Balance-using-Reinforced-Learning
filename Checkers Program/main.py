'''
Input ee ee twice to quit the game
ee ee를 두 번 입력하여 게임 종료
'''

# Gets the move from the User
def getUserMove(b, color):
    #white = 0, black = 1
    #백 = 0, 흑 = 1
    statement1 = "Player " + str(color+1) + " Select one of the following moves: "
    
    #Whites can only go up(negative) and blacks can only go down(positive)
    #백은 위로(음수), 흑은 아래로(양수)만 이동 가능
    direction = 2 * color - 1
    
    print(statement1)
    
    #Print the available moves
    #가능한 움직임들 출력
    cnt1 = 0
    available = b.Obtain(((-1, direction), (1, direction)), color)
    for i in available:
        cnt2 = 0
        for j in i:
            if cnt1 != 0 and cnt2 == 0:
                print(', ', end = '')
            if cnt2 > 0:
                print(' ->', end = ' ')
            print(chr(j[1] + 97) + str(j[0]), end = '')
            cnt2 += 1
        cnt1 += 1
    print('\n')
    
    
    while True:

        move = []
        move = input().lower().split()
        
        if (len(move) != 2) or len(move[0]) != 2 or len(move[1]) != 2:
            print ("That is not a valid move, try again.", statement1)
            continue
            
        #Force Shutdown
        #강제종료
        if move[0][1] == 'e':
            b.force = 'shutdown'
            return
        
        
        moveFromTup = (int(move[0][1]), ord(move[0][0]) - 97)
        moveToTup = (int(move[1][1]), ord(move[1][0]) - 97)
        
        # Is the piece one we own?
        #이 말은 우리 것인가?
        if (moveFromTup not in b.piecelist[color]) and (moveFromTup not in b.piecelist[color+2]):
            print ("You do not own", moveFromTup, "please select one of.", b.piecelist[color], "or", b.piecelist[color+2])
            continue
        
        move = [moveFromTup, moveToTup]
        
        tester = []
        for i in available:
            tester.append([i[0], i[-1]])
            
        #is the move a capable move?
        #이 움직임은 가능한가?
        if move not in tester:
            print("Invalid move. Please select one of the moves above.")
            continue

        for i in available:
            if move[0] == i[0] and move[-1] == i[-1]:
                move = i
                break
        
        if moveFromTup in b.piecelist[color]:
            b.move(move, color)
            
        else:
            b.move(move, color+2)
        
        b.promote(moveToTup, color)
        
        b.printBoard()
        return
        
### MAIN PROGRAM ###

width = 8
height = 8

b = board(width, height)
b.printBoard()
print("Welcome to checkers.")

# Main game loop
while b.force == 0:
    
    #Player 1's Turn
    #플레이어 1 턴
    getUserMove(b, 0)
    
    #Player 2's Turn
    #플레이어 2 턴  
    getUserMove(b, 1)
    
    
    if len(b.piecelist[1]) == 0 and len(b.piecelist[3]) == 0:
        print("White Wins\nGame Over")
        break
    elif len(b.piecelist[0]) == 0 and len(b.piecelist[2]) == 0:
        print("Black Wins\nGame Over")
        break
