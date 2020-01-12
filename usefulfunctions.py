# initialize 4 prime numbers preferablly very big ones rng, x , m , r ,z
rng = 5915587277
def random(x,m,r,z):                           #x,rand m should be rediculously big prime numbers,z is the larget number you want out of it
  global rngout
  global rng
  a = (x * rng) + (x  * m)
  rng = a % r
  rngout = rng % z
                                                  #rngout is the random number produced(note  r is a second seed)

def decing(x,m,r,z,k):                                 #z is the number of zeros you want after the decimal place of the samll number,x,r and m should be rediculously big prime numbers,k is significant digits
  global deciout
  global rngout
  deciout = "0."
  for x in range(z):
    deciout = deciout + str(0)
  for y in range(k - z):
    random(x,m,r,10)
    deciout = deciout + str(rngout)
  deciout = float(deciout)
