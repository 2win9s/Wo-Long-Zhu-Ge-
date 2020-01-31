import numpy as np
import time as tm


P1 = " "     #defignes all the varibles i'll use.
P2 = " "
P3 = " "
P4 = " "
P5 = " "
P6 = " "
P7 = " "
P8 = " "
P9 = " "
Turn = "X"
Winner = " "
tst = 0
TurnCount = 0
WinConditMet = 0

print ("welcome to tic tac toe")   #some intro stuff
print ("X what's your first move?")
Move = int(input())          # sets "move" to an interger the player imputs

def CheckWin():        #checks if the game has been won
  global Winner
  global WinConditMet
  if P1 != " ":    #checks if P1 or the first slot is not blank
    Winner = P1    #changes the var winner from blank to whatever is in P1
    if P1 == P2 == P3:   #checks if the top row is all the same
      WinConditMet = 1   # changes WinConditMet to 1 or true
    if P1 == P5 == P9:   #rinse and repeat
      WinConditMet = 1
    if P1 == P4 == P7:
      WinConditMet = 1

  elif P5 != " ":
    Winner = P5
    if P4 == P5 == P6:
      WinConditMet = 1
    if P2 == P5 == P8:
      WinConditMet = 1
    if P3 == P5 == P7:
      WinConditMet = 1

  elif P9 != " ":
    Winner = P9
    if P7 == P8 == P9:
      WinConditMet = 1
    if P3 == P6 == P9:
      WinConditMet = 1

def HasWon():               #checks if someone has won yet
  if TurnCount >3:          #checks if greater than 4 turns have been made
    CheckWin()              #runs "CheckWin"
    global WinConditMet
    global Winner
    if WinConditMet == 1:  #checks if WinConditMet is true
      if Winner == "X":    #checks if x won
        print ("X Wins!")
        exit()             #kills the game
      elif Winner == "O":  #checks if o won
        print ("O wins!")
        exit()             #kills the game

def MakeMove():            #is called to make a move
  global P1                #these are all set as globals so that when the yare referanced later they are not local because python just makes a new varible rather than scanning for global ones by default. this tells it scan for the global value
  global P2
  global P3
  global P4
  global P5
  global P6
  global P7
  global P8
  global P9
  global Turn
  global Move
  global tst 
  global TurnCount

  def DispBoard():          #prints out the board
    global TurnCount
    print (P1, "¦", P2,"¦", P3) #prints the varible P1 followed by ¦ and so on.
    print ("---------")
    print (P4, "¦", P5,"¦", P6)
    print ("---------")  
    print (P7, "¦", P8,"¦", P9)
    HasWon()                #calls the "HasWon" function made earlier.
    print ("What's your next move?")
    TurnCount += 1          #increases the turncount by 1 at the end of each move for the HasWon function

  if Move is 1:             #Checks if the players latest input is "1"
    if P1 == " ":           #checks slot 1 is blank
      if Turn == "X":         #checks who's turn it is
        P1 = "X"            #sets the varble "P1"
        Turn = "O"            #changes the turn
      else:
        P1 = "O"
        Turn = "X"
      DispBoard()
      tst = 1
    elif P1 != " ":
      if tst == 0:
        print ("That space is already taken! Try again!")
        Move = int(input())
        tst = 1
      else:
        Move = int(input())
        tst = 0
        return MakeMove()
        
  elif Move is 2:
    if P2 == " ":
      if Turn == "X":
        P2 = "X"
        Turn = "O"
      else:
        P2 = "O"
        Turn = "X"
      DispBoard()
      tst = 1
    elif P2 != " ":
      if tst == 0:
        print ("That space is already taken! Try again!")
        Move = int(input())
        tst = 1
      else:
        Move = int(input())
        tst = 0
        return MakeMove()
        
  elif Move is 3:
    if P3 == " ":
      if Turn == "X":
        P3 = "X"
        Turn = "O"
      else:
        P3 = "O"
        Turn = "X"
      DispBoard()
      tst = 1
    elif P3 != " ":
      if tst == 0:
        print ("That space is already taken! Try again!")
        Move = int(input())
        tst = 1
      else:
        Move = int(input())
        tst = 0
        return MakeMove()
        
  elif Move is 4:
    if P4 == " ":
      if Turn == "X":
        P4 = "X"
        Turn = "O"
      else:
        P4 = "O"
        Turn = "X"
      DispBoard()
      tst = 1
    elif P4 != " ":
      if tst == 0:
        print ("That space is already taken! Try again!")
        Move = int(input())
        tst = 1
      else:
        Move = int(input())
        tst = 0
        return MakeMove()
      
  elif Move is 5:
    if P5 == " ":
      if Turn == "X":
        P5 = "X"
        Turn = "O"
      else:
        P5 = "O"
        Turn = "X"
      DispBoard()
      tst = 1
    elif P5 != " ":
      if tst == 0:
        print ("That space is already taken! Try again!")
        Move = int(input())
        tst = 1
      else:
        Move = int(input())
        tst = 0
        return MakeMove()

  elif Move is 6:
    if P6 == " ":
      if Turn == "X":
        P6 = "X"
        Turn = "O"
      else:
        P6 = "O"
        Turn = "X"
      DispBoard()
      tst = 1
    elif P6 != " ":
      if tst == 0:
        print ("That space is already taken! Try again!")
        Move = int(input())
        tst = 1
      else:
        Move = int(input())
        tst = 0
        return MakeMove()
      
  elif Move is 7:
    if P7 == " ":
      if Turn == "X":
        P7 = "X"
        Turn = "O"
      else:
        P7 = "O"
        Turn = "X"
      DispBoard()
      tst = 1
    elif P7 != " ":
      if tst == 0:
        print ("That space is already taken! Try again!")
        Move = int(input())
        tst = 1
      else:
        Move = int(input())
        tst = 0
        return MakeMove()
      
  elif Move is 8:
    if P8 == " ":
      if Turn == "X":
        P8 = "X"
        Turn = "O"
      else:
        P8 = "O"
        Turn = "X"
      DispBoard()
      tst = 1
    elif P8 != " ":
      if tst == 0:
        print ("That space is already taken! Try again!")
        Move = int(input())
        tst = 1
      else:
        Move = int(input())
        tst = 0
        return MakeMove()
      
  elif Move is 9:
    if P9 == " ":
      if Turn == "X":
        P9 = "X"
        Turn = "O"
      else:
        P9 = "O"
        Turn = "X"
      DispBoard()
      tst = 1
    elif P9 != " ":
      if tst == 0:
        print ("That space is already taken! Try again!")
        Move = int(input())
        tst = 1
      else:
        Move = int(input())
        tst = 0
        return MakeMove()
      
  else:
    print ("That's an invalid move. Please use number 1 ~ 9.")
    Move = int(input())


for TurnCount in range(0, 9):
  MakeMove()
