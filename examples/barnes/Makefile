CXX = g++
CXXFLAGS += -std=c++17 -Wall -Wunused-result -O3 -DNDEBUG

cuda:
	$(CXX) $(CXXFLAGS) -o cuda.out Main.cpp ../../accelerator/cuda/libcuda.a -I../../include -L/usr/local/cuda/lib64 -lcudart
