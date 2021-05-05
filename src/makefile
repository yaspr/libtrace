CC=gcc

CFLAGS=-ggdb -Wall -Wextra

LFLAGS=-shared -fPIC -ldl

OFLAGS=-march=native -mtune=native -Ofast

all: libtrace.so

libtrace.so: libtrace.c
	$(CC) $(CFLAGS) $(LFLAGS) $(OFLAGS) $< -o $@

clean:
	rm -Rf libtrace.so
