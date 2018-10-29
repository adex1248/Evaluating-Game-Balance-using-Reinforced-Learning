hands = {'left':0, 'right':1}

def getMove(game, turn):
    print("Player" + str(turn + 1) + "'s turn")
    while True:
        a = input("Please input the type of action: 0 for combining and 1 for attacking")
        try:
            a = int(a)
        except:
            print('Invalid, try again')
            continue
        if a == 2:
            game.force = 2
            return
        elif a == 0:
            b = input("How much do you want to have in your left hand?")
            try:
                b = int(b)
            except:
                print('Invalid, try again')
                continue
            if b > np.sum(game.po[turn]) or b > 5 or b < 0:
                print('Invalid, try again')
                continue
            elif b == game.po[turn][1] or b == game.po[turn][0]:
                print("That move is impossible, try again")
                continue
            else:
                game.combine(turn, b)
                return
        
        elif a == 1:
            b = input("Which hand do you want to use to attack?")
            if b != "left" and b != "right":
                print("Invalid, try again")
                continue
            c = input("Which hand do you want to attack?")
            if c != "left" and c != "right":
                print("Invalid, try again")
                continue
          
            b = hands[b]
            c = hands[c]
            
            if game.po[turn][b] > 0 and game.po[1 - turn][c] > 0:
                game.attack(turn, b, c)
                return
            else:
                print("Invalid, try again")
                continue
            
            
          
        print(game.po)
        
game = chop()
print("Game Start!")
print(game.po)
          
while True:
    getMove(game, 0)
    print(game.po)
    
    if np.sum(game.po[1]) == 0:
        print('player 1 wins!')
        break
    if np.sum(game.po[0]) == 0:
        print('player 2 wins!')
        break
        
    getMove(game, 1)
    print(game.po)
          
    if np.sum(game.po[1]) == 0:
        print('player 1 wins!')
        break
    if np.sum(game.po[0]) == 0:
        print('player 2 wins!')
        break
        
    if game.force == 2:
        break
