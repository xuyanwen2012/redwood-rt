NVCC = "$(shell which nvcc)"

ARCH = -arch=sm_53
# Toucan 'sm_75', Goose 'sm_53'
# ARCH = -arch=sm_75

NVCCFLAGS   += -std=c++14 -O3 -Xptxas -O3 -lnvToolsExt 
NVCCFLAGS   += $(ARCH)

MODULES   := . kernels
SRC_DIR   := $(addprefix ./,$(MODULES))
CU_SRC     := $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.cu))

OBJS := $(patsubst %.cu,%.o,$(CU_SRC))

all: $(OBJS)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -I../../include -c $<

clean:
	rm -rf $(OBJS) *.o

.PHONY: clean
