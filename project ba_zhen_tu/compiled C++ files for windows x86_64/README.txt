These .exe files were compiled with gcc 9.3 with the flags -O3 -static.

In our code the openMP parallel regions typically use proc_bind(spread), all openMP for loops use default scheduling and we also use openMP synchronisation hints.

If you want more flexibility change/ edit and compile the source code with gcc, also remember to download and link the necessary libraries
