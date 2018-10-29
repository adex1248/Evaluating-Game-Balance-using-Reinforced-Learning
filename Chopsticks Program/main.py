import numpy as np

po = np.ones((2,2))
comp = ['0','1']

while True:
    while 1:
        a = input('p1 type of action')
        b = input('property')
        if a == 1 and b < np.sum(po[0]) and b <= 5:
            po[0] = np.array([int(b), np.sum(po[0]) - int(b)])
            if po[0][0] == 5:
                po[0][0] == 0
            if po[0][1] == 5:
                po[0][1] == 0
            print(po)
        elif a == 1:
            print('error')
            continue
        elif a == 2:
            b1 = str(b)[0]
            b2 = str(b)[1]
            if b1 in comp and b2 in comp:
                po[1][int(b2) - 1] += po[0][int(b1) - 1]
                if po[1][int(b2) - 1] >= 5:
                    po[1][int(b2) - 1] = 0
                print(po)
            else:
                print('error')
                continue
        else:
            print('error')
            continue
        break
        
    if np.sum(po[1]) == 0:
        print('p1 victory')
        break
    if np.sum(po[0]) == 0:
        print('p2 victory')
        break
        
    while 1:
        a = input('p2 type of action')
        b = input('property')
        if a == 1 and b < np.sum(po[1]) and b <= 5:
            po[1] = np.array([int(b), np.sum(po[1]) - int(b)])
            if po[1][0] == 5:
                po[1][0] == 0
            if po[1][1] == 5:
                po[1][1] == 0
            print(po)
        elif a == 1:
            print('error')
            continue
        elif a == 2:
            b1 = str(b)[0]
            b2 = str(b)[1]
            if b1 in comp and b2 in comp:
                po[0][int(b2) - 1] += po[0][int(b1) - 1]
                if po[0][int(b2) - 1] >= 5:
                    po[0][int(b2) - 1] = 0
                print(po)
            else:
                print('error')
                continue
        else:
            print('error')
            continue
        break
    
    if np.sum(po[1]) == 0:
        print('p1 victory')
        break
    if np.sum(po[0]) == 0:
        print('p2 victory')
        break
