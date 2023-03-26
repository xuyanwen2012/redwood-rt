include ../../Makefile.inc

cuda: Main.o 
	$(CXX) Main.o Redwood/cuda/libcuda.a -L/usr/local/cuda/lib64 -lcudart -o cuda.out

Main.o: Main.cpp
	$(CXX) $(CXXFLAGS) -c Main.cpp -o Main.o

clean:
	rm *.out *.o *.a