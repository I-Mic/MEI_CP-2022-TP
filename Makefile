CC = gcc
BIN = bin/
SRC = src/
INCLUDES = include/
EXEC = k_means -lm

#CFLAGS = -funroll-loops -O2
CFLAGS = -ftree-vectorize -msse4 -O2

.DEFAULT_GOAL = k_means

k_means: $(SRC)k_means.c $(BIN)utils.o
	$(CC) $(CFLAGS) $(SRC)k_means.c $(BIN)utils.o -o $(BIN)$(EXEC)

$(BIN)utils.o: $(SRC)utils.c $(INCLUDES)utils.h
	$(CC) $(CFLAGS) -c $(SRC)utils.c -o $(BIN)utils.o

clean:
	rm -r bin/*

run:
	time ./$(BIN)$(EXEC)
