#Use icc or icx for Intel compilers 
CC=gcc

CFLAGS=-ggdb -Wall -Wextra -fno-omit-frame-pointer -rdynamic

#Use -mkl with Intel compilers
LFLAGS=-lcblas

OFLAGS=-march=native -mtune=native -Ofast -funroll-loops -flto

gemm: gemm.c
	$(CC) $(CFLAGS) $(OFLAGS) $< -o $@ $(LFLAGS)

test:
	LD_PRELOAD=../src/libtrace.so ./gemm 100 10

clean:
	rm -Rf gemm

