rng = 5202642720986189087034837832337828472969800910926501361967872059486045713145450116712488685004691423
def random(x,m,z):                           #x and m should be rediculously big prime numbers,z is the larget number you want out of it
  global rngout
  global rng
  a = (x * rng) + (x  * m)
  rng = a % m
  rngout = rng % z
                                                  #rngout is the random number produced(note  r is a second seed)
