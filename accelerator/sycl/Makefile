DPCPP = clang++ -fsycl

DPCPPFLAGS += -std=c++17 -Wall -Wunused-result -O3 -DNDEBUG

MODULES   := . kernels
SRC_DIR   := $(addprefix ./,$(MODULES))
SRC     := $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.cpp))

OBJS := $(patsubst %.cpp,%.o,$(SRC))

all: $(OBJS)

%.o: %.cpp
	$(DPCPP) $(DPCPPFLAGS) -I../../include -c $<

clean:
	rm -rf *.o

.PHONY: clean
