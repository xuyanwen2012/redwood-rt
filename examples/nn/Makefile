include ../../Makefile.inc

REDWOOD_CUDA_LIB := -L ../../accelerator/cuda -lredwoodcuda

SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

all: cuda

cuda: $(OBJECTS)
	$(CXX) -o cuda.out $(OBJECTS) $(REDWOOD_CUDA_LIB) -L /usr/local/cuda/lib64 -lcudart -fopenmp

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $(SOURCES) -I ../../include -fopenmp

clean:
	rm -f $(OBJECTS) *.out 