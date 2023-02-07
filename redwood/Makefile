include ../Makefile.inc

CUDA_DIR := ../accelerator/cuda
CU_OBJ   := $(wildcard $(CUDA_DIR)/*.o)

SYCL_DIR := ../accelerator/sycl
SYCL_OBJ   := $(wildcard $(SYCL_DIR)/*.o)

CPU_DIR := ../accelerator/cpu
CPU_OBJ   := $(wildcard $(CPU_DIR)/*.o)

# LIB_OBJ := $(wildcard *.o)

# all: sycl cuda

cpu: RedwoodNn.o
	ar rcs ../lib/RedwoodCpu.a RedwoodNn.o $(CPU_OBJ)

sycl: RedwoodNn.o
	ar rcs ../lib/RedwoodSycl.a RedwoodNn.o $(SYCL_OBJ)

cuda: RedwoodNn.o
	ar rcs ../lib/RedwoodCuda.a RedwoodNn.o $(CU_OBJ)

# sycl_bh: RedwoodBh.o
# 	ar rcs ../lib/RedwoodSyclBh.a RedwoodBh.o ./sycl/UsmAlloc.o ./sycl/Kernel.o ./sycl/Barnes.o

# cuda_bh: RedwoodBh.o
# 	ar rcs ../lib/RedwoodCudaBh.a RedwoodBh.o ./cuda/UsmAlloc.o ./cuda/Kernel.o ./cuda/Barnes.o

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -I../include -c $< 

clean:
	rm -rf *.o
