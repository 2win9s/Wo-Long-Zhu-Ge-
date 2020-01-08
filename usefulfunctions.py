def random(x,m):                                   #x is the seed can be any pseudo random number,dice the time in milliseconds etc.(recommend a prime number),m is maximum value from rng(recommend a prime number)
  global rngout
  a = (x * rngout) + (x  * m)
  rngout = a % m
                                                  #rngout is the random number produced(note  r is a second seed)
