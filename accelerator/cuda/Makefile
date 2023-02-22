NVCC = $(shell which nvcc)
NVCCFLAGS = -std=c++14 -O3 -Xptxas -O3 

INCLUDES := -I../../include/
INCLUDES += -I./extern/cub-1.17.2/ -I./extern/thrust-1.17.2/

# Toucan 'sm_75', Goose 'sm_53'
# ARCH = -arch=sm_75
ARCH = -arch=sm_53

SOURCES = $(wildcard *.cu)
OBJECTS = $(SOURCES:.cu=.o)

LIBRARY = libcuda.a

all: $(LIBRARY)

$(LIBRARY): $(OBJECTS)
	ar rcs $(LIBRARY) $(OBJECTS)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(ARCH) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(LIBRARY)
