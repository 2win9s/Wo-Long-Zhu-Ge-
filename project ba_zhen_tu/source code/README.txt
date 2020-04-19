WE HAVE NO GPU/ACCELERATOR OFFLOADING IN OUR CODE
The code in this folder requires the following libraries:
Boost
OpenMP
and C++ standard library

The code also must be compiled with the newest version of gcc or a compiler that implements both C++ threading and openMP with POSIX threads. 

This is becuase that we have put some variables in thread_local storage where the openMP threadprivate clause wouldn't allow us to do so. 

If you can get the threadprivate clause to work for these variables with your compiler then feel free to use threadprivate instead and ignore the previous 2 sentences.

In our code the openMP parallel regions typically use proc_bind(spread), all openMP for loops use default scheduling and we also use openMP synchronisation hints(only in openMP 5 and above).

Feel free to mess around with the code before compiling to suit your needs.
