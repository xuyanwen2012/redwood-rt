include ../../Makefile.inc

SRC     := $(wildcard *.cpp)
OBJS := $(patsubst %.cpp,%.o,$(SRC))

LIBRARY = libcpu.a

all: $(LIBRARY)

$(LIBRARY): $(OBJS)
	ar rcs $(LIBRARY) $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -I../../include -c $<

clean:
	rm -rf *.o $(LIBRARY)

