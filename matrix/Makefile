CC = gcc
CFLAGS = -Wall -g

main: main.c mat.c
	$(CC) $(CFLAGS) main.c mat.c rand.c -o main

matlab: matlab2.c mat.c rand.c
	$(CC) $(CFLAGS) matlab2.c mat.c rand.c -o matlab2
