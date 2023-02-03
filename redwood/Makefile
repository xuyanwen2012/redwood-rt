include ../Makefile.inc

OBJ := $(wildcard *.o)

all: sycl cuda

sycl: Redwood.o 
	ar rcs ../lib/RedwoodSycl.a Redwood.o ./sycl/UsmAlloc.o ./sycl/Kernel.o

cuda: Redwood.o 
	ar rcs ../lib/RedwoodCuda.a Redwood.o ./cuda/UsmAlloc.o ./cuda/Kernel.o 

Redwood.o:
	$(CXX) $(CXXFLAGS) -c Redwood.cpp 

clean:
	rm -rf *.o
