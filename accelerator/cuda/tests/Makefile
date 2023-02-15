NVCC = $(shell which nvcc)
NVCCFLAGS = -std=c++14 -O3 -Xptxas -O3 -I../extern/cub-1.17.2/ -I../extern/thrust-1.17.2/

ARCH = -arch=sm_53

SOURCES = $(wildcard *.cu)

all: 
	$(NVCC) $(NVCCFLAGS) $(ARCH) $(SOURCES) 
