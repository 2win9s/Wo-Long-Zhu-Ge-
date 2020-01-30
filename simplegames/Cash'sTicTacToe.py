import numpy as np
import time as tm
print ("welcome to tic tac toe")

Move = int(input())
global P1
global P2
global P3
global P4
global P5
global P6
global P7
global P8
global P9
P1 = " "
P2 = " "
P3 = " "
P4 = " "
P5 = " "
P6 = " "
P7 = " "
P8 = " "
P9 = " "
global Turn
Turn = 0


def MakeMove():
  def DispBoard():
    print (P1, "¦", P2,"¦", P3)
    print ("---------")
    print (P4, "¦", P5,"¦", P6)
    print ("---------")  
    print (P7, "¦", P8,"¦", P9)

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
    print ("slot 2")
  elif Move is 3:
    print ("slot 3")
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
