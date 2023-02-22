include ../../Makefile.inc

CUDA_LIB := ../../accelerator/cuda/libcuda.a
SYCL_LIB := ../../accelerator/sycl/libsycl.a
CPU_LIB := ../../accelerator/cpu/libcpu.a

SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

all: cuda sycl

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $(SOURCES) -I../../include

cpu: $(OBJECTS)
	$(CXX) -o  cpu.out $(OBJECTS) $(CPU_LIB)

cuda: $(OBJECTS)
	$(CXX) -o  cuda.out $(OBJECTS) $(CUDA_LIB) -L/usr/local/cuda/lib64 -lcudart

sycl: $(OBJECTS)
	clang++ -fsycl -o sycl.out $(OBJECTS) $(SYCL_LIB) -L$(SYCL_LIB_DIR) -lsycl

clean:
	rm -f $(OBJECTS) *.out