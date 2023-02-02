NVCC = "$(shell which nvcc)"

# Toucan 'sm_75', Goose 'sm_53'
ARCH = -arch=sm_75

NVCCFLAGS   += -std=c++14 -O3 -Xptxas -O3 -lnvToolsExt 
NVCCFLAGS   += $(ARCH)

CU_SRC := $(wildcard *.cu)
OBJS := $(patsubst %.cu,%.o,$(CU_SRC))

all: $(OBJS)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $<

clean:
	rm -rf *.o 
