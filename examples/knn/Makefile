include ../../Makefile.inc

CUDA_LIB := ../../accelerator/cuda/libcuda.a

SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

all: cuda

cuda: $(OBJECTS)
	$(CXX) -o cuda.out $(OBJECTS) $(CUDA_LIB) -L/usr/local/cuda/lib64 -lcudart

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $(SOURCES) -I../../include

clean:
	rm -f $(OBJECTS) *.out 