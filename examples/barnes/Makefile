include ../../Makefile.inc

cuda:
	$(CXX) $(CXXFLAGS) -o cuda.out Main.cpp ../../accelerator/cuda/libcuda.a -I../../include -L/usr/local/cuda/lib64 -lcudart

sycl: 
# Sycl must use DPCPP to link the final binary.
	$(CXX) $(CXXFLAGS) -c Main.cpp -I../../include
	clang++ -fsycl -o sycl.out Main.o ../../accelerator/sycl/libsycl.a -L$(SYCL_LIB_DIR) -lsycl
