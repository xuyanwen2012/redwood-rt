include ../../Makefile.inc

CPU_LIB := ../../accelerator/cpu/libcpu.a

SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

all: cpu

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $(SOURCES) -I../../include

cpu: $(OBJECTS)
	$(CXX) -o  cpu.out $(OBJECTS) $(CPU_LIB)

clean:
	rm -f $(OBJECTS) *.out