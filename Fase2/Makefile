CC = gcc
BIN = bin/
SRC = src/
INCLUDES = include/
EXEC = k_means
THREADS = 8
CP_CLUSTERS = 4

#CFLAGS = -funroll-loops -O2
CFLAGS = -ftree-vectorize -msse4 -O2 -fopenmp

.DEFAULT_GOAL = k_means

k_means: $(SRC)k_means.c $(BIN)utils.o
	$(CC) $(CFLAGS) $(SRC)k_means.c $(BIN)utils.o -o $(BIN)$(EXEC) -lm

$(BIN)utils.o: $(SRC)utils.c $(INCLUDES)utils.h
	$(CC) $(CFLAGS) -c $(SRC)utils.c -o $(BIN)utils.o

clean:
	rm -r bin/*
	
runseq:
	./$(BIN)$(EXEC) 10000000 $(CP_CLUSTERS)
	
runpar:
	./$(BIN)$(EXEC) 10000000 $(CP_CLUSTERS) $(THREADS)
