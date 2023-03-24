include ../../Makefile.inc

NVCC = $(shell which nvcc)
NVCCFLAGS = -std=c++14 -O3 -Xptxas -O3 --expt-relaxed-constexpr

INCLUDES += -I../../accelerator/cuda/extern/cub-1.17.2/ -I../../accelerator/cuda/extern/thrust-1.17.2/

ARCH = -arch=sm_53

cpu: Main.o libcpu.a
	$(CXX) Main.o libcpu.a -o cpu.out

cuda: Main.o libcuda.a
	$(CXX) Main.o libcuda.a -L/usr/local/cuda/lib64 -lcudart -o cuda.out

Main.o: Main.cpp
	$(CXX) $(CXXFLAGS) -c Main.cpp -o Main.o

cpu.o: Redwood/Cpu.cpp
	$(CXX) $(CXXFLAGS) -c  Redwood/Cpu.cpp -o cpu.o

cuda.o: Redwood/Cuda.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(ARCH) -c ./Redwood/Cuda.cu -o cuda.o
	
libcpu.a: cpu.o
	ar rcs libcpu.a cpu.o

libcuda.a: cuda.o
	ar rcs libcuda.a cuda.o

clean:
	rm *.out *.o *.a