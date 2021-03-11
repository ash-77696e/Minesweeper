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

def markSafe(agent, board, safeList, moves, knowledge_base):
    while len(safeList) > 0:
        currSafe = safeList.pop()
        print('make a safe move at')
        print(currSafe)
        x, y = currSafe
        agent[x][y] = board[x][y]
        # add to knowledge base since we make a new move
        knowledge_base.append((currSafe, agent[x][y], 
        get_safe_neighbors(agent, currSafe), 
        get_mine_neighbors(agent, currSafe), get_hidden_neighbors(agent, currSafe)))
        moves.remove(currSafe)

def basic_agent(board):
    agent = np.zeros((board.shape[0], board.shape[0]), int)
    dimension = board.shape[0]
    agent[:] = -1 # -1 represents an unknown cell

    defused = 0

    moves = []
    knowledge_base = []

    for x in range(agent.shape[0]):
        for y in range(agent.shape[0]):
            moves.append((x, y))

    while len(moves) > 0:
        # Update knowledge base with most updated information before making a move
        knowledge_base = [(kb[0], kb[1], get_safe_neighbors(agent, kb[0]), 
        get_mine_neighbors(agent, kb[0]), get_hidden_neighbors(agent, kb[0])) for kb in knowledge_base]

        if len(knowledge_base) > 0:
            removedItems = []
            moveMade = False

            for kb in knowledge_base:
                coord = kb[0]
                clue = kb[1]
                safe = kb[2]
                mines = kb[3]
                hidden = kb[4]

                if clue - mines == hidden:
                    newMines = get_all_hidden_neighbors(agent, coord)
                    for coord in newMines:
                        moves.remove(coord)
                        agent[coord[0]][coord[1]] = 9
                        defused += 1
                    
                    moveMade = True
                    removedItems.append(kb)
                    continue
                
                # corner cell
                if (x == 0 and y == 0) or (x == 0 and y == dimension - 1) or (x == dimension - 1 and y == 0) or (x == dimension - 1 and y == dimension - 1):
                    if (3 - clue) - safe == hidden:
                        safeList = get_all_hidden_neighbors(agent, coord)
                        markSafe(agent, board, safeList, moves, knowledge_base) # make a bunch of safe moves
                        moveMade = True
                        removedItems.append(kb)
                        continue
                
                # border cell
                elif x == 0 or y == 0:
                    if (5 - clue) - safe == hidden:
                        safeList = get_all_hidden_neighbors(agent, coord)
                        markSafe(agent, board, safeList, moves, knowledge_base) # make a bunch of safe moves
                        moveMade = True
                        removedItems.append(kb)
                        continue

                else:
                    if (8 - clue) - safe == hidden:
                        safeList = get_all_hidden_neighbors(agent, coord)
                        markSafe(agent, board, safeList, moves, knowledge_base)
                        moveMade = True
                        removedItems.append(kb)
                        continue

            if moveMade == True:
                for item in removedItems:
                    knowledge_base.remove(item)
                continue # Since we have made a move(s) through our basic inference, no need to pick a random move

            # if we don't need to remove anything from the knowledge base, that means we didn't make any moves through basic inference
            # so, we must make a random choice


        # pick random spot from moves list
        i = randint(0, len(moves) - 1)
        coord = moves[i]
        x, y = coord
        agent[x][y] = board[x][y]

        # if we didn't pick a mine, update knowledge base
        if agent[x][y] != 9:
            print('random safe move at')
            print(coord)
            knowledge_base.append((coord, agent[x][y], 
            get_safe_neighbors(agent, coord), 
            get_mine_neighbors(agent, coord), get_hidden_neighbors(agent, coord)))
        else:
            print('you picked a mine (random) at:')
            print((x, y))
        
        moves.remove(coord)
    
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

if __name__ == '__main__':
    board = generate_board(16, 60)
    basic_agent(board)