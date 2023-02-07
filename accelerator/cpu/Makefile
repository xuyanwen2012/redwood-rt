include ../../Makefile.inc

MODULES   := . kernels
SRC_DIR   := $(addprefix ./,$(MODULES))
SRC     := $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.cpp))

OBJS := $(patsubst %.cpp,%.o,$(SRC))

all: $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -I../../include -c $<

clean:
	rm -rf *.o

.PHONY: clean
