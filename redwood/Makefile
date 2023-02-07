include ../Makefile.inc

CUDA_DIR := ../accelerator/cuda
CU_OBJ   := $(wildcard $(CUDA_DIR)/*.o)

LIB_OBJ := $(wildcard *.o)

all: sycl cuda

# sycl: RedwoodNn.o
# 	ar rcs ../lib/RedwoodSycl.a RedwoodNn.o ./sycl/UsmAlloc.o ./sycl/Kernel.o ./sycl/Nn.o

cuda: $(LIB_OBJ)
	ar rcs ../lib/RedwoodCuda.a $(LIB_OBJ) $(CU_OBJ)

# sycl_bh: RedwoodBh.o
# 	ar rcs ../lib/RedwoodSyclBh.a RedwoodBh.o ./sycl/UsmAlloc.o ./sycl/Kernel.o ./sycl/Barnes.o

# cuda_bh: RedwoodBh.o
# 	ar rcs ../lib/RedwoodCudaBh.a RedwoodBh.o ./cuda/UsmAlloc.o ./cuda/Kernel.o ./cuda/Barnes.o

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -I../include -c $< 

clean:
	rm -rf *.o
