import numpy as np
import sys
from random import *


'''
This python file is used to play out a game of minesweeper on a randomly generated board with either a basic agent or and advanced
agent. The basic agent uses weak inference algorithms to solve the board, whereas the advanced agent combines multiple clue values and
knowledge known about cells on the board to more effectively identify and defuse mines. To play a game of minesweeper the user must
first generate a board using generate_board() with the specified dimension and density they want. Then, they can pass this board into 
either the basic_agent() or advanced_agent() functions depending on which type of algorithm they want to play out minesweeper. When
the board is generated the total number of mines on the board is returned, and when one of the agents is done solving the board it 
returns the amount of mines that were successfully defused on the board. These values can be used to find the success rate of the 
algorithm in its playthrough of minesweeper on a randomly generated board.

Authors: Ashwin Haridas, Ritin Nair
'''

'''
This function is used to generate a board of size dimension x dimension that has the specified mine density. The mines are randomly
assigned to cells on the board.

Input: dimension of maze, mine density
Output: randomly generated board based on specified mine density, total number of mines on the board

'''

def generate_board(dimension, density):
    board = np.zeros((dimension, dimension), dtype=int)
    totalMines = 0

    # This loop assigns the mines to random cells on the board based on the specified mine density
    for x in range(board.shape[0]):
        for y in range(board.shape[0]):
            if random() < density:
                board[x][y] = 9
                totalMines += 1
    
    # for each cell on the board, the number of neighbors that are mines are summed up and assigned as that specific cell's
    # clue value
    for x in range(0, dimension):
        for y in range(0, dimension):

            # skip assigning a clue value for a cell that has a mine
            if board[x][y] == 9:
                continue
            
            # otherwise, count the total number of neighbors a cell has that are mines
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

            # assign the cell its clue value
            board[x][y] = mines
    
    return board, totalMines

'''
This function takes in a list of coordinates assumed to be safe based on inferences and reveals them on the agent board. The cells
are then assigned clue values based on the board matrix after being revealed. After a cell has been revealed, it is also removed from
the moves list so that it is not chosen to be revealed at a later point in time.

Input: the current agent board, the board matrix which contains the clue values of all cells, a list of safe cells to reveal on the
       agent board, the moves list which contains cells that still could possibly be revealed on the agent board, and the knowledge 
       base
Output: There is not direct returned value from this function, but the agent board, moves list, and knowledge base are all modified
'''

def markSafe(agent, board, safeList, moves, knowledge_base):
    while len(safeList) > 0:
        currSafe = safeList.pop()
        print('make a safe move at')
        print(currSafe)
        x, y = currSafe
        # reveal cell in the safe list
        agent[x][y] = board[x][y]

        # add to knowledge base since we make a new move
        knowledge_base.append(currSafe)
        # once we make a move we should remove it from moves list
        if currSafe in moves:
            moves.remove(currSafe)

'''
This function takes a list of cells known to be mines based on inferences made about the board. It reveals these cells and defuses the 
mines on them and also removes these cells from the moves list so they are not selected to be revealed randomly in the future.

Input: The agent board, the list of mines to defuse, and the moves list
Output: The amount of mines defused and the total mines defused 
'''
def markMines(agent, newMines, moves):
    defused = 0
    total = 0
    for coord in newMines:
        if coord in moves:
            moves.remove(coord)
            print('mark mine at')
            print(coord)
        x, y = coord
        agent[x][y] = 9
        defused += 1
        total += 1
    
    return defused, total

'''
This function is the basic agent which solves the minesweeper board with weak inference. Each cell in the 
knowledge base is iterated through and certain conditions are checked to see if any inferences can be made about each of these cells'
hidden neighbors. These inferences are only based on information about each cell individually and do not combine multiple clues
in the knowledge base together. Therefore, this algorithm uses weak inferences. If there are no inferences that can be made 
based on the information about cells in the knowledge base, then a random cell is selected to be revealed. If a mine is revealed
due to an inference then the defused count is incremented to award the player a point. However, if it is revealed based on random 
selection and not inference then the mine is said to explode, but the game continues regardless. The game goes until every cell in the
board is revealed.

Input: board which contains all the mine locations and clue values for the cells
Output: the amount of mines successfully defused
'''
def basic_agent(board):
    agent = np.zeros((board.shape[0], board.shape[0]), int)
    dimension = board.shape[0]
    agent[:] = -1 # -1 represents an unknown cell and initially every cell in the board is hidden

    defused = 0
    total = 0

    moves = [] # list of all cells that still need to be revealed and are valid choices to be chosen
    
    knowledge_base = [] # list of cells with known clues that can be used to get information and make inferences

    # add all cells to the moves list because they are all initially hidden
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
                # extract information from a cell in the knowledge base
                # the information is calculated at each step to ensure that the knowledge base is up to date
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
                        
                # regular case
                else:
                    if (8 - clue) - safe == hidden:
                        safeList = get_all_hidden_neighbors(agent, coord)
                        markSafe(agent, board, safeList, moves, knowledge_base)
                        inferenceMade = True
                        moveMade = True
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
    
    #print(agent)
    #print(knowledge_base)
    return defused

'''
This function is the advanced agent that plays minesweeper on a randomly generated board. It uses strong inference in its algorithm
and combines multiple clues when possible to make inferences about the board that the basic agent is not able to. It uses the basic 
agent algorithm to make basic inferences about cells in the knowledge base first, and then it combines all the clues in the knowledge
base together to see how they interact and if they reveal any additional information. These advanced inferences help the advanced agent
detect a higher amount of mines than the basic agent which gives it a higher success rate when playing minesweeper.

Input: the board with the mine locations and clue values
Output: the amount of mines defused
'''
def advanced_agent(board):
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
        print(agent)

        # try to make a move through basic inference
        if len(knowledge_base) > 0:
            removedItems = []
            inferenceMade = False
            moveMade = False
            advancedInfer = False

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
                        
                # regular case
                else:
                    if (8 - clue) - safe == hidden:
                        safeList = get_all_hidden_neighbors(agent, coord)
                        markSafe(agent, board, safeList, moves, knowledge_base)
                        inferenceMade = True
                        moveMade = True
                        removedItems.append(coord)
                        
            # a basic inference was able to be made
            if inferenceMade == True:
                for item in removedItems:
                    if item in knowledge_base:
                        knowledge_base.remove(item)
            
            else:
                colToCoordList = {} # dictionary which has the column number as the key and the cell as the value
                coordToCol = {} # dictionary which has the cell as the key and the column number as the value

                count = 0
                for coord in knowledge_base:
                    hiddenList = get_all_hidden_neighbors(agent, coord)
                    for neighbor in hiddenList:
                        if neighbor not in colToCoordList.values():
                            colToCoordList[count] = neighbor
                            coordToCol[neighbor] = count
                            count += 1

                matrix = [] # this matrix acts as a representation of all the combined clues and information in the knowledge base
                            # this is referred to in the write up as the knowledge matrix

                # list of equations
                #print(knowledge_base)
                for coord in knowledge_base:
                    x, y = coord
                    clue = agent[x][y]
                    safe = get_safe_neighbors(agent, coord)
                    mines = get_mine_neighbors(agent, coord)
                    hidden = get_hidden_neighbors(agent, coord)
                
                    equation = [] # each row in the matrix is an equation
                    hiddenList = get_all_hidden_neighbors(agent, coord)

                    # the equation will initally have 0 or 1 values for every column but the last
                    # a 0 means the cell that the corresponding column represents is not a neighbor of the cell the row with the
                    # equation is for, and a 1 means that the corresponding column represents is a neighbor of
                    # the cell the row with the equation is for
                    # the last value is the amount of mines the cell the row is for has left in its hidden neighbors
                    # each row represents and equation, and the columns with 1 values represent the variables involved in the equation
                    # the variables are essentially the remaining hidden neighbors of a cell
                    # these variables can be assigned a value of either 0 (safe) or 1 (mine)
                    for i in range(0, count):
                        if colToCoordList[i] in hiddenList:
                            equation.append(1)
                        else:  
                            equation.append(0)
                    equation.append(clue - mines) # add last column value to the equation
                    matrix.append(equation) # add the row to the matrix
                
                reduce_matrix(matrix) # use Gaussian Elimination to simplify the matrix and attempt to isolate variables to solve them
                # use the siimplified matrix and additional available information to make advanced inferences about the board
                tempDefused, tempTotal, advancedInfer, advancedMove = infer_from_matrix(matrix, agent, board, moves, knowledge_base, colToCoordList) 
                defused += tempDefused
                total += tempTotal

                removedItems = []

                for coord in knowledge_base:
                    if get_hidden_neighbors(agent, coord) == 0:
                        removedItems.append(coord)
                
                for coord in removedItems:
                    knowledge_base.remove(coord)
                
                if advancedInfer:
                    print('inference was made !!')

            if inferenceMade or advancedInfer:
                continue # Since we have made a move(s) through our basic / advanced inference, no need to pick a random move

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
            print('random safe move at')
            print(coord)
            knowledge_base.append(coord)
        else:
            total += 1
            print('you picked a mine (random) at:')
            print(coord)
        
        moves.remove(coord)
    
    #print(agent)
    #print(knowledge_base)
    print(agent)
    return defused

'''
This function uses Gaussian Elimination to reduce the knowledge matrix that is passed in as an argument. This simplified matrix is 
used at a later step of the algorithm to make more advanced inferences about the board.

Input: The knowledge matrix, or the matrix with the system of linear equations based on the clues in the knowledge base
Output: There is no direct return, but the matrix passed in is simplified for later use

'''
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
            row_swap(matrix, i, swapRowNum) # we want the leading entry of the row to have the highest absolute value
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

'''
This functions swaps two rows in a matrix

Input: the matrix and the two rows to be swapped
Output: no direct return, but the matrix passed into the function will have its rows swapped
'''
def row_swap(matrix, row1, row2):
    colDim = len(matrix[0])
    tempRow = []
    
    for j in range(0, colDim):
        tempRow.append(matrix[row1][j])
    
    for j in range(0, colDim):
        matrix[row1][j] = matrix[row2][j]
    
    for j in range(0, colDim):
        matrix[row2][j] = tempRow[j]

'''
This function uses the reduced knowledge matrix to make advanced inferences about cells on the board

Input: reduced matrix, agent, board, moves, knowledge_base, colToCoordList
Output: defused mines, total mines defused, if an inference was made, if a move was made
'''

def infer_from_matrix(matrix, agent, board, moves, knowledge_base, colToCoordList):
    rowDim  = len(matrix)
    colDim = len(matrix[0])
    newMines = []
    safeList = []

    for i in range(0, rowDim):
        # if a row has all 0s, 1s, and -1s (except for the last column) and the last column equals the number of 1s in the row
        if ones_zeros_negatives(matrix, i) and matrix[i][colDim - 1] == count_ones(matrix, i):
            for j in range(0, colDim - 1):
                if matrix[i][j] == 1: # 1s in the row are mines
                    newMines.append(colToCoordList[j])
                if matrix[i][j] == -1: # -1s in the row are safe
                    safeList.append(colToCoordList[j])
        # if a row has all 0s and 1s (except for the last column) and the last column equals 0
        elif ones_zeros(matrix, i) and matrix[i][colDim - 1] == 0:
            for j in range(0, colDim - 1):
                if matrix[i][j] == 1: # mark the hidden neighbors as safe
                    safeList.append(colToCoordList[j])
    
    newMinesLen = len(newMines)
    safeMinesLen = len(safeList)
    defused, total = markMines(agent, newMines, moves)
    markSafe(agent, board, safeList, moves, knowledge_base)

    if safeMinesLen > 0:
        return defused, total, True, True

    if newMinesLen > 0:
        return defused, total, True, False

    return defused, total, False, False

'''
This function checks if a row is all 0s, 1s, and -1s

Input: matrix and row number
Output: True or False

'''
def ones_zeros_negatives(matrix, rowNum): 
    colDim = len(matrix[0])
    for j in range(0, colDim - 1):
        if matrix[rowNum][j] != 0 and matrix[rowNum][j] != 1 and matrix[rowNum][j] != -1:
            return False
    return True

'''
This function checks if a row is all 0s and 1s

Input: matrix and row number
Output: True or False
'''
def ones_zeros(matrix, rowNum):
    colDim = len(matrix[0])
    for j in range(0, colDim - 1):
        if matrix[rowNum][j] != 0 and matrix[rowNum][j] != 1:
            return False
    return True

'''
This function counts the amount of ones in a row (except for the last column)

Input: matrix and row number
Output: number of ones
'''
def count_ones(matrix, rowNum): 
    count = 0
    colDim = len(matrix[0])
    for j in range(0, colDim - 1):
        if matrix[rowNum][j] == 1:
            count += 1
    return count
'''
This functions gets all the hidden neighbors of a cell and returns the list of hidden neighbors

Input: agent board and cell
Output: list of hidden neighbors
'''
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

'''
This function returns the number of safe neighbors a cell has

Input: agent board and cell
Output: number of safe neighbors
'''
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

'''
This function returns the number of mine neighbors a cell has

Input: agent board and cell
Output: number of mine neighbors
'''
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

'''
This function returns the number of hidden neighbors a cell has

Input: agent board and cell
Output: number of hidden neighbors
'''
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

'''
This function is used to run trials and find the average success rate of the basic agent over 50 trials

Input: dimension and density of the board
Output: average success rate over 50 trials
'''
def run_basic_trials(dim, density):
    average = 0
    for i in range(50):
        board, totalMines = generate_board(dim, density)
        defused = basic_agent(board)
        average += (defused / totalMines)
    
    return (average / 50)

'''
This function is used to run trials and find the average success rate of the advanced agent over 50 trials

Input: dimension and density of the board
Output: average success rate over 50 trials
'''
def run_advanced_trials(dim, density):
    average = 0
    for i in range(50):
        board, totalMines = generate_board(dim, density)
        defused = advanced_agent(board)
        average += (defused / totalMines)

    return (average / 50)

'''
This function is the main function
'''
if __name__ == '__main__':
    basic_average = run_basic_trials(25, 0.4)
    advanced_average = run_advanced_trials(25, 0.4)
    print(basic_average)
    print(advanced_average)

    # board, totalMines = generate_board(40, 0.3)
    # defused = basic_agent(board)
    # print(totalMines)
    # print(defused)
