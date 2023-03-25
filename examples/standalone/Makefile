include ../../Makefile.inc

NVCC = $(shell which nvcc)
NVCCFLAGS = -std=c++14 -O3 -Xptxas -O3 --expt-relaxed-constexpr

INCLUDES += -I../../accelerator/cuda/extern/cub-1.17.2/ -I../../accelerator/cuda/extern/thrust-1.17.2/

ARCH = -arch=sm_53

debug: MainDebug.o libcuda.a
	$(CXX) MainDebug.o libcuda.a -L/usr/local/cuda/lib64 -lcudart -o cuda.out

cuda: Main.o libcuda.a
	$(CXX) Main.o libcuda.a -L/usr/local/cuda/lib64 -lcudart -o cuda.out

MainDebug.o: MainDebug.cpp
	$(CXX) $(CXXFLAGS) -c MainDebug.cpp -o MainDebug.o

Main.o: Main.cpp
	$(CXX) $(CXXFLAGS) -c Main.cpp -o Main.o

cuda.o: Redwood/Cuda.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(ARCH) -c ./Redwood/Cuda.cu -o cuda.o
	
libcuda.a: cuda.o
	ar rcs libcuda.a cuda.o

clean:
	rm *.out *.o *.a