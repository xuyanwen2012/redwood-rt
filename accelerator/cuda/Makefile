NVCC = $(shell which nvcc)
NVCCFLAGS = -std=c++14 -O3 -Xptxas -O3 --expt-relaxed-constexpr

INCLUDES += -I ./extern/cub-1.17.2/ -I ./extern/thrust-1.17.2/

SOURCES = $(wildcard *.cu)
OBJECTS = $(SOURCES:.cu=.o)

all: libredwoodcuda.a

%.o: %.cu
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) -c $< -o $@ -I ../../include/

libredwoodcuda.a: $(OBJECTS)
	ar rcs $@ $^

clean:
	rm -f *.o libredwoodcuda.a
