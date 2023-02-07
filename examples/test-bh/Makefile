include ../../Makefile.inc

SYCL_LIB_DIR := /home/tsorensen/sycl_workspace/llvm/build/lib

all: sycl

sycl: 
# Sycl must use DPCPP to link the final binary.
	$(CXX) $(CXXFLAGS) -c Main.cpp -I../../include
	clang++ -fsycl -o sycl.out Main.o ../../lib/RedwoodSycl.a -L$(SYCL_LIB_DIR) -lsycl

clean:
	rm -rf *.o *.out