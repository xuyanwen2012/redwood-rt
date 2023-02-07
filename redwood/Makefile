include ../Makefile.inc

OBJ := $(wildcard *.o)

all: sycl cuda

sycl: RedwoodNn.o
	ar rcs ../lib/RedwoodSycl.a RedwoodNn.o ./sycl/UsmAlloc.o ./sycl/Kernel.o ./sycl/Nn.o

cuda: RedwoodNn.o
	ar rcs ../lib/RedwoodCuda.a RedwoodNn.o ./cuda/UsmAlloc.o ./cuda/Kernel.o ./cuda/Nn.o

sycl_bh: RedwoodBh.o
	ar rcs ../lib/RedwoodSyclBh.a RedwoodBh.o ./sycl/UsmAlloc.o ./sycl/Kernel.o ./sycl/Barnes.o

cuda_bh: RedwoodBh.o
	ar rcs ../lib/RedwoodCudaBh.a RedwoodBh.o ./cuda/UsmAlloc.o ./cuda/Kernel.o ./cuda/Barnes.o

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -rf *.o
