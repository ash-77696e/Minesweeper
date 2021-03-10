import numpy as np
import sys
from random import *

def generate_board(dimension, mines):
    board = np.zeros((dimension, dimension), dtype=int)

    while mines > 0:
        x = randint(0, dimension - 1)
        y = randint(0, dimension - 1)

        if board[x][y] == 9:
            continue

        board[x][y] = 9
        mines -= 1
    
    for x in range(0, dimension):
        for y in range(0, dimension):
            if board[x][y] == 9:
                continue

            mines = 0

            if x - 1 >= 0:
                if board[x - 1][y] == 9:
                    mines += 1
                if y - 1 >= 0:
                    if board[x - 1][y - 1] == 9:
                        mines += 1
                if y + 1 < dimension:
                    if board[x - 1][y + 1] == 9:
                        mines += 1
            
            if x + 1 < dimension:
                if board[x + 1][y] == 9:
                    mines += 1
                if y - 1 >= 0:
                    if board[x + 1][y - 1] == 9:
                        mines += 1
                if y + 1 < dimension:
                    if board[x + 1][y + 1] == 9:
                        mines += 1
            
            if y - 1 >= 0:
                if board[x][y - 1] == 9:
                    mines += 1
            
            if y + 1 < dimension:
                if board[x][y + 1] == 9:
                    mines += 1

            board[x][y] = mines
    
    return board

def basic_agent(board):
    agent = np.zeros((board.shape[0], board.shape[0]), int)
    dimension = board.shape[0]
    agent[:] = -1 # -1 represents an unknown cell

    defused = 0

    all_moves = []
    knowledge_base = []
    safeList = []

    for x in range(agent.shape[0]):
        for y in range(agent.shape[0]):
            all_moves.append((x, y))

    while len(all_moves) > 0:
        knowledge_base = [(kb[0], kb[1], get_safe_neighbors(agent, kb[0]), 
        get_mine_neighbors(agent, kb[0]), get_hidden_neighbors(agent, kb[0])) for kb in knowledge_base]

        # if there is something in our safe list, pick it as your move and move on
        if len(safeList) > 0:
            currSafe = safeList.pop()

            if currSafe not in all_moves:
                continue
            
            x, y = currSafe

            agent[x][y] = board[x][y]

            knowledge_base.append((currSafe, agent[x][y], 
            get_safe_neighbors(agent, currSafe), 
            get_mine_neighbors(agent, currSafe), get_hidden_neighbors(agent, currSafe)))

            all_moves.remove(currSafe)
            
            continue

        if len(knowledge_base) > 0:
            removedItem = None
            remove = False

            for kb in knowledge_base:
                coord = kb[0]
                clue = kb[1]
                safe = kb[2]
                mines = kb[3]
                hidden = kb[4]

                if clue - mines == hidden:
                    newMines = get_all_hidden_neighbors(agent, coord)
                    for coord in newMines:
                        if coord in all_moves:
                            all_moves.remove(coord)
                            agent[coord[0]][coord[1]] = 9
                            defused += 1
                    
                    remove = True
                    removedItem = kb
                    break
                
                # corner cell
                if (x == 0 and y == 0) or (x == 0 and y == dimension - 1) or (x == dimension - 1 and y == 0) or (x == dimension - 1 and y == dimension - 1):
                    if (3 - clue) - safe == hidden:
                        newSafe = get_all_hidden_neighbors(agent, coord)
                        safeList += newSafe
                        remove = True
                        removedItem = kb
                        break
                
                # border cell
                elif x == 0 or y == 0:
                    if (5 - clue) - safe == hidden:
                        newSafe = get_all_hidden_neighbors(agent, coord)
                        safeList += newSafe
                        remove = True
                        removedItem = kb
                        break

                else:
                    if (8 - clue) - safe == hidden:
                        newSafe = get_all_hidden_neighbors(agent, coord)
                        safeList += newSafe
                        remove = True
                        removedItem = kb
                        break

            if remove == True:
                knowledge_base.remove(removedItem)
                continue # we only skip the random spot if we deduce some new information


        # pick random spot from moves list
        i = randint(0, len(all_moves) - 1)
        coord = all_moves[i]
        x, y = coord
        agent[x][y] = board[x][y]

        # if we didn't pick a mine, update knowledge base
        if agent[x][y] != 9:
            knowledge_base.append((coord, agent[x][y], 
            get_safe_neighbors(agent, coord), 
            get_mine_neighbors(agent, coord), get_hidden_neighbors(agent, coord)))
        else:
            print('you picked a mine at:')
            print((x, y))
        
        all_moves.remove(all_moves[i])
    
    print(agent)
    print(defused)
        
def get_all_hidden_neighbors(agent, coord):
    neighbors = []
    dimension = agent.shape[0]
    x, y = coord

    if x - 1 >= 0:
        if agent[x - 1][y] == -1:
            neighbors.append((x - 1, y))
        if y - 1 >= 0:
            if agent[x - 1][y - 1] == -1:
                neighbors.append((x - 1, y - 1))
        if y + 1 < dimension:
            if agent[x - 1][y + 1] == -1:
                neighbors.append((x - 1, y + 1))
        
    if x + 1 < dimension:
        if agent[x + 1][y] == -1:
            neighbors.append((x + 1, y))
        if y - 1 >= 0:
            if agent[x + 1][y - 1] == -1:
                neighbors.append((x + 1, y - 1))
        if y + 1 < dimension:
            if agent[x + 1][y + 1] == -1:
                neighbors.append((x + 1, y + 1))
    
    if y - 1 >= 0:
        if agent[x][y - 1] == -1:
            neighbors.append((x, y - 1))
    
    if y + 1 < dimension:
        if agent[x][y + 1] == -1:
            neighbors.append((x, y + 1))
    
    return neighbors

def get_safe_neighbors(agent, coord):
    safe = 0
    dimension = agent.shape[0]
    x, y = coord

    if x - 1 >= 0:
        if agent[x - 1][y] != 9 and agent[x - 1][y] != -1:
            safe += 1
        if y - 1 >= 0:
            if agent[x - 1][y - 1] != 9 and agent[x - 1][y - 1] != -1:
                safe += 1
        if y + 1 < dimension:
            if agent[x - 1][y + 1] != 9 and agent[x - 1][y + 1] != -1:
                safe += 1
    
    if x + 1 < dimension:
        if agent[x + 1][y] != 9 and agent[x + 1][y] != -1:
            safe += 1
        if y - 1 >= 0:
            if agent[x + 1][y - 1] != 9 and agent[x + 1][y - 1] != -1:
                safe += 1
        if y + 1 < dimension:
            if agent[x + 1][y + 1] != 9 and agent[x + 1][y + 1] != -1:
                safe += 1
    
    if y - 1 >= 0:
        if agent[x][y - 1] != 9 and agent[x][y - 1] != -1:
            safe += 1
    
    if y + 1 < dimension:
        if agent[x][y + 1] != 9 and agent[x][y + 1] != -1:
            safe += 1

    return safe

def get_mine_neighbors(agent, coord):
    mines = 0
    dimension = agent.shape[0]
    x, y = coord

    if x - 1 >= 0:
        if agent[x - 1][y] == 9:
            mines += 1
        if y - 1 >= 0:
            if agent[x - 1][y - 1] == 9:
                mines += 1
        if y + 1 < dimension:
            if agent[x - 1][y + 1] == 9:
                mines += 1
    
    if x + 1 < dimension:
        if agent[x + 1][y] == 9:
            mines += 1
        if y - 1 >= 0:
            if agent[x + 1][y - 1] == 9:
                mines += 1
        if y + 1 < dimension:
            if agent[x + 1][y + 1] == 9:
                mines += 1
    
    if y - 1 >= 0:
        if agent[x][y - 1] == 9:
            mines += 1
    
    if y + 1 < dimension:
        if agent[x][y + 1] == 9:
            mines += 1

    return mines

def get_hidden_neighbors(agent, coord):
    hidden = 0
    dimension = agent.shape[0]
    x, y = coord

    if x - 1 >= 0:
        if agent[x - 1][y] == -1:
            hidden += 1
        if y - 1 >= 0:
            if agent[x - 1][y - 1] == -1:
                hidden += 1
        if y + 1 < dimension:
            if agent[x - 1][y + 1] == -1:
                hidden += 1
    
    if x + 1 < dimension:
        if agent[x + 1][y] == -1:
            hidden += 1
        if y - 1 >= 0:
            if agent[x + 1][y - 1] == -1:
                hidden += 1
        if y + 1 < dimension:
            if agent[x + 1][y + 1] == -1:
                hidden += 1
    
    if y - 1 >= 0:
        if agent[x][y - 1] == -1:
            hidden += 1
    
    if y + 1 < dimension:
        if agent[x][y + 1] == -1:
            hidden += 1

    return hidden

board = generate_board(16, 90)
basic_agent(board)