CC = g++
CFLAGS = -std=c++11 -Wall -Wextra -Wpedantic
LDFLAGS =

all: KnnSet

KnnSet: KnnSet.o
	$(CC) $(LDFLAGS) $^ -o $@

KnnSet.o: KnnSet.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o KnnSet
