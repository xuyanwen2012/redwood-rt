include ../Makefile.inc

all:
	$(CXX) $(CXXFLAGS) -c Redwood.cpp 
	ar rcs ../lib/RedwoodCuda.a Redwood.o ./cuda/UsmAlloc.o ./cuda/Kernel.o 

clean:
	rm -rf *.o
