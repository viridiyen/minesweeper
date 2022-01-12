
import numpy as np 
import random
import sympy as sp
from fractions import Fraction

class MineBoard:
    board = None #2D array of 0 or 1

    def __init__(self, dim: int, num_mines: int):
        self.generate_board(dim, num_mines) 
        self.dim = dim
        self.num_mines = num_mines
        
    """
    Method Purpose: Generate a square mineboard given the dimension and the number of mines
    Spae Complexity: O(n^2), n= dim
    Time Complexity: O(n^2), n=dim
    2D array of values 0, 1 (1 = mine)
    """
    def generate_board(self, dim: int, num_mines: int):
        #generate 2D array of size dim x dim of zeros 
        self.board = [ [0] * dim for _ in range(dim)]
        #populate randomly with num_mines number of mines, represented by int 1
        xy_used = []
        while num_mines > 0:        
            #O(1): generate a random cell location
            x = random.randint(0, dim-1)    
            y = random.randint(0, dim-1)
            #if the cell location is not already picked...
            if (x,y) not in xy_used:    #O(n): n=num_mines: if the set location is not already picked...
                xy_used.append((x,y))   #O(1): add the cell location to the list of cell locations with mines
                self.board[x][y] = 1    #O(1): lookup time in a 2d array: set the cell in the board to a one (a mine exists there)
                num_mines-=1            #O(1): decrease the number of mines (should be 0 at final)
                
    '''
    Method Purpose: Calculate the clue from the cells surrounding neighbours
    Space Complexity:  O(1)
    Time Complexity:   O(1)
    return whether its a mine (1 means mine, 0 means no mine)
    calculate clue (-1 if there was a mine at that location)
    '''
    def query(self, cell: (int,int)):
        r = cell[0]
        c = cell[1]
        isMine = (self.board[r][c] == 1)    

        if not isMine:
            #calculate the clue by adding all 8 neighbors
            tuples = [(r+1, c),         #below
                        (r+1, c-1),     #bottom left
                        (r+1, c+1),     #bottom right
                        (r, c+1),       #right
                        (r, c-1),       #left
                        (r-1, c),       #above
                        (r-1, c-1),     #upper left
                        (r-1, c+1)]     #upper right
            clue = 0 

            #O(c), at max there will be 8 neighbours to calculate the clue
            for tup in tuples:  
                if tup[0] >= 0 and tup[0] < self.dim and tup[1] >= 0 and tup[1] < self.dim:
                    clue += self.board[tup[0]][tup[1]]
        else: 
            clue = -1

        return isMine, clue #isMine:0,1 : clue: 0 to 8, inclusive

#Cell class that will contain necessary information for a cell update
class Cell:
    def __init__(self):
        self.status = 'unknown'     #types of statuses: safe (0), mine(1), unknown
        self.unknown_neighbors=8    #the number of unknown neighbors around the cell
        self.mines_identified=0     #the number of mines around the cell that have been identified
        self.safe_identified = 0    #the number of safe cells around the cell that have been identified
        self.clue = -1

class Agent(): 
    cell_array = None #2D array of Cell's, makes up the knowledge base 

    def __init__(self, dim, num_mines, verbose):
        self.dim = dim
        self.num_mines = num_mines
        self.board = MineBoard(dim, num_mines)
        self.init_KB()
        self.verbose = verbose
    
    """
    Method Purpose: print the current state of the knowledge base so that users can see the progression of the game when they run play()
    Time Complexity: O(n^2)
    """
    def print_KB(self):
        for i in range(self.dim):
            display = ""
            for j in range(self.dim):
                if self.cell_array[i][j].status=="safe" and self.cell_array[i][j].clue != -1:
                    display+=str(self.cell_array[i][j].clue)
                    display+="  "
                elif self.cell_array[i][j].status=="safe":
                    display+='\x1b[0;36;42m'+ "S  " + '\x1b[0m' #colors the cell green if its identified as safe but still unrevealed
                elif self.cell_array[i][j].status=="mine":      
                    display+="M  "
                else:                                           
                    display+='\x1b[0;34;40m'+ "?  " + '\x1b[0m' #colors the cell grey if it is completely unknown
            print(display)
    
    
    """
    Method Purpose: Initialize a blank knowledge base so that play() can fill it
    Space Complexity: O(dim^2)
    Time Complexity: O(dim^2)
    """
    def init_KB(self):
        self.cell_array = [[Cell() for i in range(self.dim)] for j in range(self.dim)] 
         
    '''
    Method Purpose: Marks the specified cell as a status (3 options: safe, mine, unknown) with 
                    an optional clue (clue can be passed None)
    Space Complexity: O(1)
    Time Complexity:  O(1), the max number of unknown neighbours is 8
    '''
    def mark(self, cell: (int, int), status, clue):

        #update the unknown_neighbors, mine_identified, safe_identified for all neighbors
        if self.cell_array[cell[0]][cell[1]].status == 'unknown':
            neighbors = self.get_neighbors([cell]) 
            for nb in neighbors:
                r,c = nb[0], nb[1]
                self.cell_array[r][c].unknown_neighbors -= 1
                if status == 'mine':
                    self.cell_array[r][c].mines_identified += 1
                if status == 'safe':
                    self.cell_array[r][c].safe_identified += 1

        #update the status and clue for that cell
        self.cell_array[cell[0]][cell[1]].status = status
        if clue is not None:   
            self.cell_array[cell[0]][cell[1]].clue = clue

    '''
    Method Purpose: Returns the in-bounds neighbours of a group of cells (excluding the cells themselves)
    Space Complexity: O(n)
    Time complexity: O(n)
    '''
    def get_neighbors(self, cells):
        neighbors_set = set()
        
        #for each cell, add its neighbors to the set
        for cell in cells:
            r,c = cell[0], cell[1]
            tuples = [(r+1, c),         #below
                        (r+1, c-1),     #down left
                        (r+1, c+1),     #down right
                        (r, c+1),       #right
                        (r, c-1),       #left
                        (r-1, c),       #above
                        (r-1, c-1),     #upper left
                        (r-1, c+1)]     #upper right
            neighbors_set.update(tuples)

        #make sure the returned list does not include out of bounds tuples or tuples from cells   
        neighbors = []
        for nb in neighbors_set:    
            if ((nb[0] >= 0 and nb[0] < self.dim and nb[1] >= 0 and nb[1] < self.dim) 
                and (nb not in cells)):
                    neighbors.append(nb)

        return neighbors

    '''
    Method Purpose: make the changes to the knowledge base going out from a cell
    Time and Space Efficiency: complexity of the algorithm depends on the number of cells eventually added to the stack.
                            It only adds cells that have been unidentified, so through the course of its runtime inside play() it at worst sees every cell only once.
    '''
    def infer_from(self, cell: (int, int)):
        mines_identified = set()
        safe_identified = set()
        apply_rules_to = self.get_neighbors([cell])
        apply_rules_to.append(cell)                 

        while len(apply_rules_to) > 0:
            crnt =  apply_rules_to.pop()     
            r,c = crnt[0], crnt[1]
            if self.cell_array[r][c].clue == -1:    
                continue
            newly_identified = []
            changed_status = False

            #apply rules to crnt (I'm assuming if one applies, the other logically can't)

            #if clue - mines_identified == unknown_neighbors, update each unknown neighbor to be a mine
            if self.cell_array[r][c].clue - self.cell_array[r][c].mines_identified == self.cell_array[r][c].unknown_neighbors:
                changed_status = True
                neighbors = self.get_neighbors([crnt])
                for nb in neighbors:
                    if self.cell_array[nb[0]][nb[1]].status == 'unknown':
                        self.mark(nb, 'mine', None)
                        mines_identified.add(nb)        
                        newly_identified.append(nb)

            #if (8-clue) - safe_identified == unknown_neighbors, update each unknown neighbor to be a safe
            if (8-self.cell_array[r][c].clue) - self.cell_array[r][c].safe_identified == self.cell_array[r][c].unknown_neighbors:
                changed_status = True
                neighbors = self.get_neighbors([crnt])
                for nb in neighbors:
                    if self.cell_array[nb[0]][nb[1]].status == 'unknown':
                        self.mark(nb, 'safe', None)
                        safe_identified.add(nb)
                        newly_identified.append(nb)

            #if rule identifies new safes or mines
            if changed_status:
                # push all of the neighbors of the safe/mine block (except crnt itself)
                neighbors = self.get_neighbors(newly_identified)
                if crnt in neighbors:
                    neighbors.remove(crnt)
                apply_rules_to.extend(neighbors)

        return mines_identified, safe_identified

    '''
    Method Purpose: play the mine board game using the class Agent 
    Space Complexity: O(dim^2)
    Time Complexity: O(dim^2)
    '''
    def play(self):
        self.init_KB()
        
        mines_triggered = 0
        all_cells = {(i,j) for i in range(self.dim) for j in range(self.dim)}
        queried = set()
        mines = set()
        safe = set()

        while len(safe) + len(mines) < self.dim*self.dim:
            #try to choose a safe unqueried cell, then if that fails, choose any unqueried cell (that's not a mine)
            choices = safe.difference(queried)
            if len(choices) == 0:
                choices = all_cells.difference(queried, mines) #because of while loop condition, choices after this line is never empty
            cell = random.choice(list(choices))
            queried.add(cell)

            #query the cell and update knowledge base
            isMine, clue = self.board.query(cell)
            if isMine:
                mines_triggered+=1
                self.mark(cell, 'mine', None)
                mines.add(cell)
            else:
                self.mark(cell, 'safe', clue)
                safe.add(cell)
            if self.verbose:
                 print("Queried " + str(cell))
                 print("Triggered a mine") if isMine else print("Found it safe with clue = " + str(clue))

            #infer from that cell
            mines_identified, safe_identified = self.infer_from(cell)
            mines.update(mines_identified)
            safe.update(safe_identified)
            if self.verbose:
                    print("After inference from " + str(cell))
                    self.print_KB()
                    print()

        score = 1 - (mines_triggered/self.num_mines)
        
        if self.verbose:
             print("Final board")
             self.print_KB()
             print(str(mines_triggered) + " mines were triggered out of " + str(self.num_mines) + " mines")
             print("The score is " + str(score))
        
        return score, self.correct()

    #compares cell_array with the board to see if play() marked everything correctly
    def correct(self):
        translation = {'safe': 0, 'mine':1, 'unknown':-1}
        mineboard = self.board.board
        for i in range(self.dim):
            for j in range(self.dim):
                status = self.cell_array[i][j].status
                if translation[status] != mineboard[i][j]:
                    return False
        return True

'''
KnowledgeBase keeps a list of equation objects made up of variable objects.
Additionally, it maintains a dictionary of all cells and their values (0, 1, or None).
In the runtime analyses, n is the number of equations and m is the number of variables in each equation
'''
class KnowledgeBase:
    
    #dictionary key are cell/variable locations as a tuple; values are value of the cell (0 = safe, 1 = mine, None = unknown)
    #equationList keeps a running list of unsolved equation objects from the board
    def __init__(self, dim):
        self.dim = dim
        self.init_varDict()
        self.equationList = list()
        
    '''
    Method Purpose: initalize the variable dictionary to its default value 
    Space Complexity:O(dim^2)
    Time complexity: O(dim^2)
    '''
    #poplutes the initial KnowledgeBase dictionary with all cells initializes all cell values to None    
    def init_varDict(self):
        self.kbdict = {}
        for j in range(self.dim):
            for i in range(self.dim):
                self.kbdict[(i,j)] = None
   
    '''
    Method Purpose: Will set the variable values in the knowledge base dictionary
    Space Complexity: O(m*n)
    Time Complexity: O(m*n)
    '''
    #sets the value 0 or 1 (None is default) of variable in kb dictionary and every equation in equationsList
    def set_var_value(self, cellLoc, val):
        #mark the value in the dictionary
        self.kbdict[cellLoc] = val
        emptyEqs = []

        #remove the variables with cellLoc from every equation and reflect the change in sum_clue when necessary
        for equation in self.equationList: 
            for var in equation.listVariables:
                   if var.loc == cellLoc:
                        if val == 1:
                            equation.sum_clue -= var.coeff
                        equation.listVariables.remove(var)
                        break #assume variable is not going to appear more than once
            if len(equation.listVariables) == 0:
                emptyEqs.append(equation)
        
        #remove equations with no variables
        for emptyEq in emptyEqs:
            if emptyEq.sum_clue != 0:
                 print("Something is wrong, variables*0 is not equal to 0")
            self.equationList.remove(emptyEq)

    '''
    Method Purpose: Goes through the equation list and checks if values can be obtained from an equation
    Space Complexity: O(m*n)
    Time Complexity: O(m*n)
    '''
    #goes through equation list and checks if values can be obtained from an equation
    #returns a list of (loc, value)
    def check_for_solved(self): 
        var_ass = set()
        for equation in self.equationList:
            coeffs = []
            for var in equation.listVariables:
                coeffs.append(var.coeff)

            if len(coeffs) == 0:
                 print("Found an empty list in check_for_solved")
                 continue #to next equation
            coeffs = np.array(coeffs)

            same_sign = True
            for coeff in coeffs:
                if coeff*coeffs[0] < 0:
                    same_sign = False
                    break

            #if variables have the same sign, and if sum_clue == 0, all the variables are safe
            if same_sign and equation.sum_clue == 0:
                for var in equation.listVariables:
                    var_ass.add((var.loc, 0))
                continue #to next equation

            #if variables have the same sign, and if sum of coeffs == sum_clue, all the variables are mines
            if same_sign and np.sum(coeffs) == equation.sum_clue:
                for var in equation.listVariables:
                    var_ass.add((var.loc, 1))
                continue #to next equation

            #if there is 1 positive variable which == sum_clue and the rest are negative, positive variable is mine, other variables are safe
            #take one positive variable
            pos_coeff = None
            for i in range(len(coeffs)):
                if coeffs[i] > 0:
                    pos_coeff = (list(coeffs).pop(i), i)
                    break
            #check if the rest are negative
            negative = True
            for coeff in coeffs:
                if coeff > 0:
                    negative = False
                    break
            #if criteria are met, insert val = 1 for the positive coeff, and val = 0 for the rest
            if negative and pos_coeff is not None and equation.sum_clue == pos_coeff[0]:
                for i in range(len(equation.listVariables)):
                    var = equation.listVariables[i]
                    if i == pos_coeff[1]:
                        var_ass.add((var.loc, 1))
                    else:
                        var_ass.add((var.loc, 0))

        return var_ass
       
    '''
    Method Purpose:     inserts an equation into the equation list given listTerms, which is a list of (coeff, loc), and sum_clue  
    Space Complexity:   O(n+m) 
    Time Complexity:    O(n)
    '''
    def insert_equation(self, listTerms, sum_clue):
        sum_clue = float(sum_clue)

        #create a list of variables
        listVar = []
        for term in listTerms:
            if term[0] == 0:
                continue
            listVar.append(Variable(term[0], term[1]))

        if len(listVar) == 0:
            return None

        #edit listVar to follow certain rules

        #1. sum_clue >= 0
        if sum_clue < 0:
            sum_clue *= -1
            for var in listVar:
                var.coeff *= -1

        # 2. all coeffs are integers
        denominators = []
        if not sum_clue.is_integer():
            denominators.append(Fraction(sum_clue).limit_denominator(10))
        for var in listVar:
            if not var.coeff.is_integer():
                denominators.append(Fraction(var.coeff).limit_denominator(10))
        if len(denominators) > 0:
            lcm = np.lcm.reduce(denominators)
        else:
            lcm = 1
        sum_clue *= lcm
        sum_clue = round(sum_clue)
        for var in listVar:
            var.coeff *= lcm
            var.coeff = round(var.coeff)

         # 3. reduced down (2X + 2Y = 2 would not be allowed)
        coeffs = [var.coeff for var in listVar]
        coeffs.append(sum_clue)
        gcd = np.gcd.reduce(coeffs)
        if gcd == 0:
            print("Ran into divide by 0 issue")
            print("Trying to insert equation " + str(listVar) + " = " + str(sum_clue))
        sum_clue /= gcd
        for var in listVar:
            var.coeff /= gcd

         # 4. no repeat variables (Y + Y = 2 would not be allowed) -- this rule is maintained by the rest of the code, so this method does not need to check for it

        #check for duplicates in equationList
        neweqn = Equation(listVar, sum_clue)
        if neweqn in self.equationList: #O(equations)
            return None
        
        #add in equation to KB's equation directory
        self.equationList.append(neweqn) 
    
    '''
    Method Purpose: takes in KnowledgeBase and finds subsets other equations with the exact same variables (same coefficients and signs) considering KB
    Unfinished and deprecated in favor of matrix_solve
    '''
    def subset_solve(self):
        #sort equations by length
        temp = self.equationList
        temp = sorted(temp, key=len)    #O(equations)
        #we check every equation except the last one to see if there are subsets 
        for i in range(0, len(temp)-2):
            #take next shortest equation, find every subset of that throughout KB
            shortest = temp[i]
            indexOfSub = []
            #for every equation 
            for j in range(i+1, len(temp)):
                equation = temp[i]
                coeffEqual = True
                #that contains the subset "shortest"
                if shortest in equation:
                    #find their indeces so we can compare their coefficients
                    for tup in equation: 
                        if tup in shortest: 
                            indexOfSub.append(equation.index(tup))
                   #find coefficient: relevant for next section and later on for replacing 
                    coefficient = equation.indexOfSub[0].coeff
                    #compare the coefficients of subset found
                    for index in indexOfSub:
                       if equation.indexOfSub[index].coeff != coefficient:
                           coeffEqual = False
                           break
                    #if coefficients are equal, we adjust that equation accordingly; else we do nothing
                    if coeffEqual == True:
                        #multiply RHS of subset by coefficient and subtract from equation's sum    
                        newsum = shortest.sum_clue * coefficient 
                        equation.sum_clue -= newsum
                        #remove subset from equation
                        for k in range(0, len(indexOfSub)-1):
                            del equation.listVariables[indexOfSub]
                            #no need to update self.equationList if python doesn't make copy on var assignment and instead just points to var

    '''
    Method Purpose: Solve the system of equations in equationList to arrive at new equations that can be inserted back into equationList. 
                    This method does not infer values on its own (i.e., it does not change kbdict or remove variables from the equations). 
                    These values are inferred from the newly modified equationList when check_for_solved is called after this method.
    Space Complexity:  O(m*n)
    Time Complexity:   O(m*n)
    '''
    def matrix_solve(self):
        #convert knowledge base into one matrix
        #do guassian elimation
        #insert equations from the RREF into knowledge base
        
        #make definitive ordering of variables
        variables = set()
        for equation in self.equationList:
            for var in equation.listVariables:
                variables.add(var.loc)
        variables = list(variables)#convert into a list to have ordering, O(variables)
        variables.sort()

        #populate the M matrix with each equation from the KB (M = [A b])
        M = np.zeros((len(self.equationList), len(variables) + 1))
        for row in range(len(self.equationList)):  #O(m*n)
            equation = self.equationList[row]
            for var in equation.listVariables:
                #look at tuple and see what index it is in the variable
                col = variables.index(var.loc)
                M[row][col] = var.coeff
            M[row][len(variables)] = equation.sum_clue

        #do guassian elimination using Matrix.rref() from sympy
        M = sp.Matrix(M)
        R,_ = M.rref()
        R = np.array(R).astype(np.float64)
        R = R[~np.all(R==0, axis=1)]        #delete zero rows

        #construct equations and insert them back into the KnowledgeBase (duplicte equations and 0 coefficients will be removed by insert_equation)
        for row in range(R.shape[0]):
            #for each of the first R.shape[1]-1 columns, append to a list of Variables
            listVar = []
            for col in range(R.shape[1]-1):
                coeff, loc = R[row][col], variables[col]
                listVar.append((coeff, loc))
            #set the sum_clue
            sum_clue = R[row][R.shape[1]-1]
            self.insert_equation(listVar, sum_clue)
    
'''
Variable has class variables loc, which represents its location on the board, and coeff, which represents its coefficient in the Equation it is going to be a part of.
'''
class Variable:

    def __init__(self, coeff, loc):
        self.loc = loc
        self.coeff = float(coeff)
    
    def __lt__(self, other):
        if self.loc[0] == other.loc[0]:
            return self.loc[1] < other.loc[1]
        else:
            return self.loc[0] < other.loc[0]

    def __eq__(self, other):
        return (self.coeff, self.loc) == (other.coeff, other.loc)

    def __repr__(self):
        return str(self.coeff) + str(self.loc)

   
'''
Equation has listVar = a list of Variable objects and sum_clue = the value the variables add up to. 
'''   
class Equation:
    
    def __init__(self, listVariables, sum_clue):
        self.listVariables = listVariables
        self.listVariables.sort() #each Equation is guarenteed to be sorted
        self.sum_clue = float(sum_clue)

    def __lt__(self, other):
        return len(self.listVariables) < len(other.listVariables)
    
    def __eq__(self, other):
        return (self.listVariables == other.listVariables and self.sum_clue == other.sum_clue)

    def __repr__(self):
        equation_str = ""
        for var in self.listVariables:
            equation_str += str(var) + " + "
        equation_str = equation_str[0:len(equation_str) - 2]
        equation_str += "= " + str(self.sum_clue)
        return equation_str

'''
ImprovedAgent was made as a child class of Agent in order to reuse the basic inference functionality of Agent. 
'''
class ImprovedAgent(Agent): 

    def __init__(self, dim, num_mines, verbose):
        super().__init__(dim, num_mines, verbose)
    
    def init_KB(self):
        super().init_KB()
        self.KB = KnowledgeBase(self.dim)

    '''
    Method Purpose: Mark both knowledge bases with a status (mine or safe), and, if available, with the clue.
                    Everytime a status/value is found, only this method is called. This ensures that both knowledge bases are coordinated.
    Space Complexity: O(1)
    Time Complexity: O(1)
    '''
    def mark(self, cell: (int, int), status, clue):
        super().mark(cell, status, clue)

        #update the new knowledge base with the status
        if status == "safe":
            self.KB.set_var_value(cell, 0)
        if status == "mine":
            self.KB.set_var_value(cell, 1)

        #if there's a clue, insert equation with the unknowns
        if clue is not None:
            neighbors = self.get_neighbors([cell])
            variables = []
            for nb in neighbors:
                if self.cell_array[nb[0]][nb[1]].status == 'unknown':
                    variables.append((1, nb))
            self.KB.insert_equation(variables,clue - self.cell_array[cell[0]][cell[1]].mines_identified)
    
    '''
    Method Purpose: Play the MineBoard to completion
    Space and Time Complexity: > O(n^2) because there is the potential to do n^2 time at every of the n^2 iterations
    '''
    def play(self):
        self.init_KB()

        mines_triggered = 0
        all_cells = {(i,j) for i in range(self.dim) for j in range(self.dim)}
        queried = set()
        mines = set()
        safe = set()

        while len(safe) + len(mines) < self.dim*self.dim:
            #choose an unqueried cell randomly (from all_cells - queried - mines)
            choices = all_cells.difference(queried, mines)
            cell = random.choice(list(choices))

            ''' fix for the bug, but not used in the report
            choices = safe.difference(queried)
            if len(choices) == 0:
                choices = all_cells.difference(queried, mines)
            cell = random.choice(list(choices))
            '''

            #while there are still safe cells to query, do basic inference
            while cell is not None:
                isMine, clue = self.board.query(cell)
                if self.verbose:
                    print("Queried " + str(cell))
                    print("Triggered a mine") if isMine else print("Found it safe with clue = " + str(clue))
                if isMine:
                    mines_triggered+=1
                    self.mark(cell, 'mine', None)
                    mines.add(cell)
                else:
                    self.mark(cell, 'safe', clue)
                    safe.add(cell)
                queried.add(cell)

                #infer from that cell using basic inference from the parent class Agent
                mines_identified,safe_identified = self.infer_from(cell)
                mines.update(mines_identified)
                safe.update(safe_identified)
                if self.verbose:
                    print("After inference from " + str(cell))
                    self.print_KB()
                    print()

                #choose another safe unqueried cell (from safe-queried)
                choices = safe.difference(queried)
                if len(choices) == 0:
                    cell = None
                else:
                    cell = random.choice(list(choices))

            #consider multiple clues at a time 
            if self.verbose:
                print("Doing more advanced inference")      
            self.KB.matrix_solve()
            var_ass_set = self.KB.check_for_solved()
            while len(var_ass_set)>0:
                if self.verbose:
                    print("Found variable assignments")
                for var_ass in var_ass_set:
                    if var_ass[1] == 0:
                        self.mark(var_ass[0], "safe", None)
                        safe.add(var_ass[0])
                    if var_ass[1] == 1:
                        self.mark(var_ass[0], "mine", None)
                        mines.add(var_ass[0])
                var_ass_set = self.KB.check_for_solved()
            if self.verbose:
                self.print_KB()
                print()
            
        score = 1 - (mines_triggered/self.num_mines)

        if self.verbose:
             print("Final board")
             self.print_KB()
             print(str(mines_triggered) + " mines were triggered out of " + str(self.num_mines) + " mines")
             print("The score is " + str(score))
        
        return score, self.correct()