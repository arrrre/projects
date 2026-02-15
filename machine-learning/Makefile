CC = gcc
CFLAGS = -Wall -Wextra -pedantic -g -O3 -mavx2

main:
	$(CC) $(CFLAGS) main.c arena.c prng.c matrix.c model.c -o main.exe

predict:
	$(CC) $(CFLAGS) predict.c arena.c prng.c matrix.c model.c -o predict.exe
