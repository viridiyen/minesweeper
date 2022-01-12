# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 19:29:51 2021

@author: Audri
"""

'''
Testing script for Minesweeper
'''
import minesweeper as m

'''
def repeatX repeats the "play"
@param atype: takes 'i' or 'b' describes either Improved or Basic Agent type play method used
rep: an integer describing how many times "play" is repeated
mines: number of mines to populate board with
@returns the average score 
'''

def repeatX(atype, rep, mines):
    #find out which agent it is 
    grid = 20
    agent = m.Agent(grid, mines, True)
    outArr = []
    summation = 0
    if atype == "i":
        improved_agent = m.ImprovedAgent(grid, mines, True)
        improved_agent.board.board = agent.board.board
        for i in range(0, rep-1):
            outArr.append(improved_agent.play())
        print("played ", rep, " times")
        for elem in outArr:
            summation = summation + elem[0]
        avgScore = summation/rep 
        
    elif atype == "b":
        basic_agent = m.Agent(grid, mines, True)
        basic_agent.board.board = agent.board.board
        for i in range(0, rep-1):
            outArr.append(basic_agent.play())
        print("played ", rep, " times")
        for elem in outArr:
            summation = summation + elem[0]
        avgScore = summation/rep 
    print(avgScore)    
    return None

