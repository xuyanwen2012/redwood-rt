include ../../Makefile.inc

all: cuda

cuda:
	$(CXX) $(CXXFLAGS) -o cuda.out Main.cpp -I../../include ../../lib/RedwoodCuda.a -L/usr/local/cuda/lib64 -lcudart

clean:
	rm -rf *.o *.out