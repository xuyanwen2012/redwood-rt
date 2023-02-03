DPCPP = clang++ -fsycl

DPCPPFLAGS += -std=c++17 -Wall -Wunused-result -O3 -DNDEBUG

SRC := $(wildcard *.cpp)
OBJS := $(patsubst %.cpp,%.o,$(SRC))

all: $(OBJS)

%.o: %.cpp
	$(DPCPP) $(DPCPPFLAGS) -c $<

clean:
	rm -rf *.o 
