import sys
sys.setrecursionlimit(4000)

# Setup variables
width = 8
height = 8

# Gets the move from the User
def getUserMove(b, color):
    statement1 = "Player " + str(color+1) + " Select one of the following moves: "
    direction = 2 * color - 1
    print(statement1)
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
        #print(' count is', cnt1, cnt2, end = ' ')
        cnt1 += 1
    print('\n')
    
    while True: # Loop until proper input
        move = []
        move = input().lower().split()
        if (len(move) != 2) or len(move[0]) != 2 or len(move[1]) != 2:
            print ("That is not a valid move, try again.", statement1)
            continue
        if move[0][1] == 'e':
            b.force = 'shutdown'
            return                                          ##########################debug
        moveFromTup = (int(move[0][1]), ord(move[0][0]) - 97)
        moveToTup = (int(move[1][1]), ord(move[1][0]) - 97)
        # Is the piece we want to move one we own?
        if (moveFromTup not in b.piecelist[color]) and (moveFromTup not in b.piecelist[color+2]):
            print ("You do not own", moveFromTup, "please select one of.", b.piecelist[color], "or", b.piecelist[color+2])
            print(color, 'color')                         ###############debug
            continue
        
        move = [moveFromTup, moveToTup]
        
        tester = []
        for i in available:
            tester.append([i[0], i[-1]])
        #print(tester, 'tester')
        if move not in tester:
            print("Invalid move. Please select one of the moves above.")
            continue

        #print(*move, 'move', color)                       #########################debug
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

b = board(width, height)
b.printBoard()
print("Welcome to checkers.")

# Main game loop
while b.force == 0:
    # First it is the users turn
    getUserMove(b, 0)
    
        
    getUserMove(b, 1)
    
    if len(b.piecelist[1]) == 0 and len(b.piecelist[3]) == 0:
        print("White Wins\nGame Over")
        break
    elif len(b.piecelist[0]) == 0 and len(b.piecelist[2]) == 0:
        print("Black Wins\nGame Over")
        print(b.gameWon)                         #############################debug
        break
