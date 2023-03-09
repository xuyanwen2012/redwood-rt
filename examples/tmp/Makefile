NVCC = $(shell which nvcc)
NVCCFLAGS = -std=c++14 
# -O3 -Xptxas -O3 

INCLUDES += -I../../accelerator/cuda/extern/cub-1.17.2/ -I../../accelerator/cuda/extern/thrust-1.17.2/

# Toucan 'sm_75', Goose 'sm_53'
# ARCH = -arch=sm_75
ARCH = -arch=sm_53

all:
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(ARCH) Main.cu

clean:
	rm -f a.out
