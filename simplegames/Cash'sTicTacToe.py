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

Move = int(input())

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
    
  def DispBoard():
    print (P1, "¦", P2,"¦", P3)
    print ("---------")
    print (P4, "¦", P5,"¦", P6)
    print ("---------")  
    print (P7, "¦", P8,"¦", P9)
    Move = int(input())

  if Move is 1:
    if P1 == " ":
      if Turn == 0:
        P1 = "X"
        Turn = 1
      else:
        P1 = "O"
        Turn == 1
      DispBoard()
  elif Move is 2:
    if P2 == " ":
      if Turn == 0:
        P2 = "X"
        Turn = 1
      else:
        P2 = "O"
        Turn == 1
      DispBoard()
  elif Move is 3:
    if P3 == " ":
      if Turn == 0:
        P3 = "X"
        Turn = 1
      else:
        P3 = "O"
        Turn == 1
      DispBoard()
  elif Move is 4:
    print ("slot 4")
  elif Move is 5:
    print ("slot 5")  
  elif Move is 6:
    print ("slot 6") 
  elif Move is 5:
    print ("slot 7") 
  elif Move is 5:
    print ("slot 8") 
  elif Move is 9:
    print ("slot 9") 
  else:
    print ("false")


MakeMove()
MakeMove()
