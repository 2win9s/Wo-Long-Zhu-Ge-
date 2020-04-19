The code in this folder requires the following libraries:
Boost
OpenMP
and C++ standard library

The code also has to be compiled with the newest version of gcc or a compiler that implements both C++ threading and openMP with POSIX threads.

In our code the parallel regions typically use proc_bind(spread), change the cpu affinity in the code if it doesn't suit your needs.
