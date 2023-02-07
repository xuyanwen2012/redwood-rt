include ../Makefile.inc

OBJ := $(wildcard *.o)

all: sycl cuda

sycl: RedwoodNn.o
	ar rcs ../lib/RedwoodSycl.a RedwoodNn.o ./sycl/UsmAlloc.o ./sycl/Kernel.o ./sycl/Nn.o ./sycl/Barnes.o

cuda: RedwoodNn.o
	ar rcs ../lib/RedwoodCuda.a RedwoodNn.o ./cuda/UsmAlloc.o ./cuda/Kernel.o 

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -rf *.o
