NVCC = $(shell which nvcc)
NVCCFLAGS := -std=c++14 
NVCCFLAGS += --expt-relaxed-constexpr
# -O2 -Xptxas -O2 

# Toucan 'sm_75', Goose 'sm_53'
# ARCH = -arch=sm_75
ARCH = -arch=sm_53

all:
	$(NVCC) $(NVCCFLAGS) $(ARCH) Main.cu

clean:
	rm -f a.out
