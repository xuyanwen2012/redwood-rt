include ../../Makefile.inc

REDWOOD_CUDA_LIB := -L ../../accelerator/cuda -lredwoodcuda

SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

all: cuda

# cpu: $(OBJECTS)
# 	$(CXX) -o  cpu.out $(OBJECTS) $(CPU_LIB)

cuda: $(OBJECTS)
	$(CXX) -o  cuda.out $(OBJECTS) $(REDWOOD_CUDA_LIB) -L/usr/local/cuda/lib64 -lcudart -fopenmp

# sycl: $(OBJECTS)
# 	clang++ -fsycl -o sycl.out $(OBJECTS) $(SYCL_LIB) -L$(SYCL_LIB_DIR) -lsycl

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $(SOURCES) -I../../include -fopenmp

clean:
	rm -f $(OBJECTS) *.out