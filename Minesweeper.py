import numpy as np
import sys
from random import *

def generate_board(dimension, density):
    board = np.zeros((dimension, dimension), dtype=int)

    # while mines > 0:
    #     x = randint(0, dimension - 1)
    #     y = randint(0, dimension - 1)

    #     if board[x][y] == 9:
    #         continue

    #     board[x][y] = 9
    #     mines -= 1

    totalMines = 0

    for x in range(board.shape[0]):
        for y in range(board.shape[0]):
            if random() < density:
                board[x][y] = 9
                totalMines += 1
    
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
    
    return board, totalMines

def markSafe(agent, board, safeList, moves, knowledge_base):
    while len(safeList) > 0:
        currSafe = safeList.pop()
        #print('make a safe move at')
        #print(currSafe)
        x, y = currSafe
        agent[x][y] = board[x][y]

        if agent[x][y] == 9:
            print('this is a mine')
        # add to knowledge base since we make a new move
        knowledge_base.append(currSafe)
        # once we make a move we should remove it from moves list
        if currSafe in moves:
            moves.remove(currSafe)

def markMines(agent, newMines, moves):
    defused = 0
    total = 0
    for coord in newMines:
        if coord in moves:
            moves.remove(coord)
        x, y = coord
        agent[x][y] = 9
        defused += 1
        total += 1
    
    return defused, total


def basic_agent(board, totalMines):
    agent = np.zeros((board.shape[0], board.shape[0]), int)
    dimension = board.shape[0]
    agent[:] = -1 # -1 represents an unknown cell

    defused = 0
    total = 0

    moves = []
    
    knowledge_base = []

    for x in range(agent.shape[0]):
        for y in range(agent.shape[0]):
            moves.append((x, y))

    while len(moves) > 0:

        # try to make a move through basic inference
        if len(knowledge_base) > 0:
            removedItems = []
            inferenceMade = False
            moveMade = False

            for coord in knowledge_base:
                x, y = coord
                clue = agent[x][y]
                safe = get_safe_neighbors(agent, coord)
                mines = get_mine_neighbors(agent, coord)
                hidden = get_hidden_neighbors(agent, coord)

                # everything around is a mine
                if clue - mines == hidden:
                    newMines = get_all_hidden_neighbors(agent, coord)
                    tempDefused, tempTotal = markMines(agent, newMines, moves)
                    defused += tempDefused
                    total += tempTotal
                    
                    # for coord in newMines:
                    #     moves.remove(coord)
                    #     x, y = coord
                    #     agent[x][y] = 9
                    #     defused += 1
                    #     total += 1
                    
                    inferenceMade = True
                    removedItems.append(coord)
                    
                
                # corner cell
                elif (x == 0 and y == 0) or (x == 0 and y == dimension - 1) or (x == dimension - 1 and y == 0) or (x == dimension - 1 and y == dimension - 1):
                    if (3 - clue) - safe == hidden:
                        safeList = get_all_hidden_neighbors(agent, coord)
                        markSafe(agent, board, safeList, moves, knowledge_base) # make a bunch of safe moves
                        inferenceMade = True
                        moveMade = True
                        removedItems.append(coord)
                        
                
                # border cell
                elif x == 0 or y == 0 or x == dimension - 1 or y == dimension - 1:
                    if (5 - clue) - safe == hidden:
                        safeList = get_all_hidden_neighbors(agent, coord)
                        markSafe(agent, board, safeList, moves, knowledge_base) # make a bunch of safe moves
                        inferenceMade = True
                        moveMade = True
                        removedItems.append(coord)
                        

                else:
                    if (8 - clue) - safe == hidden:
                        safeList = get_all_hidden_neighbors(agent, coord)
                        markSafe(agent, board, safeList, moves, knowledge_base)
                        inferenceMade = True
                        modeMade = True
                        removedItems.append(coord)
                        

            if inferenceMade == True:
                for item in removedItems:
                    if item in knowledge_base:
                        knowledge_base.remove(item)

            if moveMade == True:
                continue # Since we have made a move(s) through our basic inference, no need to pick a random move

            # if we don't need to remove anything from the knowledge base, that means we didn't make any moves through basic inference
            # so, we must make a random choice

        # once we run out of moves, we end
        if len(moves) <= 0:
            break
        
        # pick random move
        i = randint(0, len(moves) - 1)
        coord = moves[i]
        x, y = coord
        agent[x][y] = board[x][y]

        # if we didn't pick a mine, add to knowledge base
        if agent[x][y] != 9:
            #print('random safe move at')
            #print(coord)
            knowledge_base.append(coord)
        else:
            total += 1
            #print('you picked a mine (random) at:')
            #print(coord)
        
        moves.remove(coord)
    
    print(agent)
    print(knowledge_base)
    return defused

def advanced_agent(board):
    '''
    agent = np.zeros((board.shape[0], board.shape[0]))
    agent[:] = -1
    print(agent)

    count = 0

    num_to_coord = {}
    coord_to_num = {}

    for x in range(agent.shape[0]):
        for y in range(agent.shape[0]):
            num_to_coord[count] = (x, y)
            coord_to_num[(x, y)] = count
            count += 1
    
    print(coord_to_num)
    print(num_to_coord)
    '''
    agent = np.zeros((board.shape[0], board.shape[0]), int)
    dimension = board.shape[0]
    agent[:] = -1 # -1 represents an unknown cell

    defused = 0
    total = 0

    moves = []
    
    knowledge_base = []

    for x in range(agent.shape[0]):
        for y in range(agent.shape[0]):
            moves.append((x, y))

    while len(moves) > 0:

        # try to make a move through basic inference
        if len(knowledge_base) > 0:
            removedItems = []
            inferenceMade = False
            moveMade = False

            for coord in knowledge_base:
                x, y = coord
                clue = agent[x][y]
                safe = get_safe_neighbors(agent, coord)
                mines = get_mine_neighbors(agent, coord)
                hidden = get_hidden_neighbors(agent, coord)

                # everything around is a mine
                if clue - mines == hidden:
                    newMines = get_all_hidden_neighbors(agent, coord)
                    tempDefused, tempTotal = markMines(agent, newMines, moves)
                    defused += tempDefused
                    total += tempTotal
                    # for coord in newMines:
                    #     moves.remove(coord)
                    #     x, y = coord
                    #     agent[x][y] = 9
                    #     defused += 1
                    #     total += 1
                    
                    inferenceMade = True
                    removedItems.append(coord)
                    
                
                # corner cell
                elif (x == 0 and y == 0) or (x == 0 and y == dimension - 1) or (x == dimension - 1 and y == 0) or (x == dimension - 1 and y == dimension - 1):
                    if (3 - clue) - safe == hidden:
                        safeList = get_all_hidden_neighbors(agent, coord)
                        markSafe(agent, board, safeList, moves, knowledge_base) # make a bunch of safe moves
                        inferenceMade = True
                        moveMade = True
                        removedItems.append(coord)
                        
                
                # border cell
                elif x == 0 or y == 0 or x == dimension - 1 or y == dimension - 1:
                    if (5 - clue) - safe == hidden:
                        safeList = get_all_hidden_neighbors(agent, coord)
                        markSafe(agent, board, safeList, moves, knowledge_base) # make a bunch of safe moves
                        inferenceMade = True
                        moveMade = True
                        removedItems.append(coord)
                        

                else:
                    if (8 - clue) - safe == hidden:
                        safeList = get_all_hidden_neighbors(agent, coord)
                        markSafe(agent, board, safeList, moves, knowledge_base)
                        inferenceMade = True
                        modeMade = True
                        removedItems.append(coord)
                        

            if inferenceMade == True:
                for item in removedItems:
                    if item in knowledge_base:
                        knowledge_base.remove(item)
            
            else:
                colToCoordList = {} 
                coordToCol = {}

                count = 0
                for coord in knowledge_base:
                    hiddenList = get_all_hidden_neighbors(agent, coord)
                    for neighbor in hiddenList:
                        if neighbor not in colToCoordList.values():
                            colToCoordList[count] = neighbor
                            coordToCol[neighbor] = count
                            count += 1

                matrix = []

                # list of equations
                print(knowledge_base)
                for coord in knowledge_base:
                    x, y = coord
                    clue = agent[x][y]
                    safe = get_safe_neighbors(agent, coord)
                    mines = get_mine_neighbors(agent, coord)
                    hidden = get_hidden_neighbors(agent, coord)
                
                    equation = []
                    hiddenList = get_all_hidden_neighbors(agent, coord)

                    for i in range(0, count):
                        if colToCoordList[i] in hiddenList:
                            equation.append(1)
                        else:  
                            equation.append(0)
                    equation.append(clue - mines)
                    matrix.append(equation)
                
                reduce_matrix(matrix)
                tempDefused, tempTotal = infer_from_matrix(matrix, agent, board, moves, knowledge_base, colToCoordList)
                defused += tempDefused
                total += tempTotal

            if moveMade == True:
                continue # Since we have made a move(s) through our basic inference, no need to pick a random move

            # if we don't need to remove anything from the knowledge base, that means we didn't make any moves through basic inference
            # so, we must make a random choice

        # once we run out of moves, we end
        if len(moves) <= 0:
            break
        
        # pick random move
        i = randint(0, len(moves) - 1)
        coord = moves[i]
        x, y = coord
        agent[x][y] = board[x][y]

        # if we didn't pick a mine, add to knowledge base
        if agent[x][y] != 9:
            #print('random safe move at')
            #print(coord)
            knowledge_base.append(coord)
        else:
            total += 1
            #print('you picked a mine (random) at:')
            #print(coord)
        
        moves.remove(coord)
    
    print(agent)
    print(knowledge_base)
    return defused

def reduce_matrix(matrix):
    rowDim = len(matrix)
    colDim = len(matrix[0])

    # forward substitution
    for j in range(0, colDim - 1):
        i = j
        if i >= rowDim:
            break
        factor = matrix[i][j]
        if factor == 0: # try row swap
            count = 1
            maxFactor = factor
            swapRowNum = i
            while(i + count < rowDim):
                if abs(matrix[i + count][j]) > maxFactor:
                    maxFactor = abs(matrix[i + count][j])
                    swapRowNum = i + count
                count += 1
            row_swap(matrix, i, swapRowNum)
            factor = matrix[i][j]
            if factor == 0:
                continue

        for x in range(0, colDim):
            matrix[i][x] = matrix[i][x] / factor
        
        rowCount = 1

        while (i + rowCount < rowDim):
            factor = matrix[i + rowCount][j] / matrix[i][j]
            for x in range(0, colDim):
                matrix[i + rowCount][x] = matrix[i + rowCount][x] - factor * matrix[i][x] 
            rowCount += 1

    # backward substitution
    if colDim > rowDim:
        j = rowDim - 1
    else:
        j = colDim - 1
    while j >= 0:
        i = j
        rowCount = 1
        while (i - rowCount >= 0):
            if matrix[i][j] == 0:
                break
            else:
                factor = matrix[i - rowCount][j] / matrix[i][j]
            for x in range(0, colDim):
                matrix[i - rowCount][x] = matrix[i - rowCount][x] - factor * matrix[i][x]
            rowCount += 1
        j -= 1

    return matrix

def row_swap(matrix, row1, row2):
    colDim = len(matrix[0])
    tempRow = []
    
    for j in range(0, colDim):
        tempRow.append(matrix[row1][j])
    
    for j in range(0, colDim):
        matrix[row1][j] = matrix[row2][j]
    
    for j in range(0, colDim):
        matrix[row2][j] = tempRow[j]
    
def infer_from_matrix(matrix, agent, board, moves, knowledge_base, colToCoordList):
    rowDim  = len(matrix)
    colDim = len(matrix[0])
    newMines = []
    safeList = []

    for i in range(0, rowDim):
        if ones_zeros_negatives(matrix, i) and matrix[i][colDim - 1] == count_ones(matrix, i):
            for j in range(0, colDim - 1):
                if matrix[i][j] == 1:
                    newMines.append(colToCoordList[j])
                if matrix[i][j] == -1:
                    safeList.append(colToCoordList[j])
        elif ones_zeros(matrix, i) and matrix[i][colDim - 1] == 0:
            for j in range(0, colDim - 1):
                if matrix[i][j] == 1:
                    safeList.append(colToCoordList[j])
    
    defused, total = markMines(agent, newMines, moves)
    markSafe(agent, board, safeList, moves, knowledge_base)
    return defused, total

def ones_zeros_negatives(matrix, rowNum): # checks if row is only 0s, 1s and -1s (except for augmented part)
    colDim = len(matrix[0])
    for j in range(0, colDim - 1):
        if matrix[rowNum][j] != 0 and matrix[rowNum][j] != 1 and matrix[rowNum][j] != -1:
            return False
    return True

def ones_zeros(matrix, rowNum):
    colDim = len(matrix[0])
    for j in range(0, colDim - 1):
        if matrix[rowNum][j] != 0 and matrix[rowNum][j] != 1:
            return False
    return True

def count_ones(matrix, rowNum): # counts ones in a row (except for the augmented part)
    count = 0
    colDim = len(matrix[0])
    for j in range(0, colDim - 1):
        if matrix[rowNum][j] == 1:
            count += 1
    return count

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
    board, totalMines = generate_board(20, 0.3)
    print(totalMines)
    defused = basic_agent(board, totalMines)
    print(defused)
    print(totalMines)
    print(defused/totalMines)

    # matrix = [[1, 0, 1, 0, 0 , 1], [0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]
    #matrix = [[1, 1, 0, 0, 1], [1, 1, 1, 0, 1], [0, 1, 1, 1, 2], [0, 0, 1, 1, 1]]
    # print('matrix before is: ')
    # print(matrix)
    # matrix = reduce_matrix(matrix)
    # print('matrix after is: ')
    # print(matrix)

    # row_swap(matrix, 1, 2)
    # print('matrix after is: ')
    # print(matrix)