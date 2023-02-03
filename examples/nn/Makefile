include ../../Makefile.inc

SYCL_LIB_DIR := /home/tsorensen/sycl_workspace/llvm/build/lib

all: cuda

sycl: 
# Sycl must use DPCPP to link the final binary.
	$(CXX) $(CXXFLAGS) -c Main.cpp -I../../include
	clang++ -fsycl -o sycl.out Main.o ../../lib/RedwoodSycl.a -L$(SYCL_LIB_DIR) -lsycl

cuda:
	$(CXX) $(CXXFLAGS) -o cuda.out Main.cpp -I../../include ../../lib/RedwoodCuda.a -L/usr/local/cuda/lib64 -lcudart

clean:
	rm -rf *.o *.out