import numpy as np
import time as tm
print ("welcome to tic tac toe")


P1 = " "
P2 = " "
P3 = " "
P4 = " "
P5 = " "
P6 = " "
P7 = " "
P8 = " "
P9 = " "
Turn = 0
GameOver = 0
Winner = " "
tst = 0
TurnCount = 4
WinConditMet = 0
Move = int(input())

def CheckWin():
  global Winner
  if P1 != " ":
    Winner = P1
    global WinConditMet
    if P1 == P2 == P3:
      WinConditMet = 1
    if P1 == P5 == P9:
      WinConditMet = 1
    if P1 == P4 == P7:
      WinConditMet = 1

def HasWon():
  if TurnCount >3:
    CheckWin()
    global WinConditMet
    global Winner
    if WinConditMet == 1:
      GameOver = 1
      if Winner == "X":
        print ("X Wins!")
        exit()
      elif Winner == "O":
        print ("O wins!")
        exit()

def MakeMove():
  global P1
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

  def DispBoard():
    global TurnCount
    print (P1, "¦", P2,"¦", P3)
    print ("---------")
    print (P4, "¦", P5,"¦", P6)
    print ("---------")  
    print (P7, "¦", P8,"¦", P9)
    #print (TurnCount)
    HasWon()
    if GameOver != 1:
      print ("Whats your next move?")
      TurnCount += 1

  if Move is 1:
    if P1 == " ":
      if Turn == 0:
        P1 = "X"
        Turn = 1
      else:
        P1 = "O"
        Turn = 0
      DispBoard()
      tst = 1
    elif P1 != " ":
      if tst == 0:
        print ("That space is already taken! Try again!")
        Move = int(input())
        tst = 1
      else:
        Move = int(input())
        return MakeMove()
        
  elif Move is 2:
    if P2 == " ":
      if Turn == 0:
        P2 = "X"
        Turn = 1
      else:
        P2 = "O"
        Turn = 0
      DispBoard()
      tst = 1
    elif P2 != " ":
      if tst == 0:
        print ("That space is already taken! Try again!")
        Move = int(input())
        tst = 1
      else:
        Move = int(input())
        return MakeMove()
        
  elif Move is 3:
    if P3 == " ":
      if Turn == 0:
        P3 = "X"
        Turn = 1
      else:
        P3 = "O"
        Turn = 0
      DispBoard()
      tst = 1
    elif P3 != " ":
      if tst == 0:
        print ("That space is already taken! Try again!")
        Move = int(input())
        tst = 1
      else:
        Move = int(input())
        return MakeMove()
        
  elif Move is 4:
    if P4 == " ":
      if Turn == 0:
        P4 = "X"
        Turn = 1
      else:
        P4 = "O"
        Turn = 0
      DispBoard()
      tst = 1
    elif P4 != " ":
      if tst == 0:
        print ("That space is already taken! Try again!")
        Move = int(input())
        tst = 1
      else:
        Move = int(input())
        return MakeMove()
      
  elif Move is 5:
    if P5 == " ":
      if Turn == 0:
        P5 = "X"
        Turn = 1
      else:
        P5 = "O"
        Turn = 0
      DispBoard()
      tst = 1
    elif P5 != " ":
      if tst == 0:
        print ("That space is already taken! Try again!")
        Move = int(input())
        tst = 1
      else:
        Move = int(input())
        return MakeMove()

  elif Move is 6:
    if P6 == " ":
      if Turn == 0:
        P6 = "X"
        Turn = 1
      else:
        P6 = "O"
        Turn = 0
      DispBoard()
      tst = 1
    elif P6 != " ":
      if tst == 0:
        print ("That space is already taken! Try again!")
        Move = int(input())
        tst = 1
      else:
        Move = int(input())
        return MakeMove()
      
  elif Move is 7:
    if P7 == " ":
      if Turn == 0:
        P7 = "X"
        Turn = 1
      else:
        P7 = "O"
        Turn = 0
      DispBoard()
      tst = 1
    elif P7 != " ":
      if tst == 0:
        print ("That space is already taken! Try again!")
        Move = int(input())
        tst = 1
      else:
        Move = int(input())
        return MakeMove()
      
  elif Move is 8:
    if P8 == " ":
      if Turn == 0:
        P8 = "X"
        Turn = 1
      else:
        P8 = "O"
        Turn = 0
      DispBoard()
      tst = 1
    elif P8 != " ":
      if tst == 0:
        print ("That space is already taken! Try again!")
        Move = int(input())
        tst = 1
      else:
        Move = int(input())
        return MakeMove()
      
  elif Move is 9:
    if P9 == " ":
      if Turn == 0:
        P9 = "X"
        Turn = 1
      else:
        P9 = "O"
        Turn = 0
      DispBoard()
      tst = 1
    elif P9 != " ":
      if tst == 0:
        print ("That space is already taken! Try again!")
        Move = int(input())
        tst = 1
      else:
        Move = int(input())
        return MakeMove()
      
  else:
    print ("false")


for x in range(0, 13):
  if GameOver == 0:
    MakeMove()
    x += 1
