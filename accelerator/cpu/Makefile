DPCPP = clang++ -fsycl
DPCPPFLAGS += -std=c++17 -Wall -Wunused-result -O3 -DNDEBUG

SRC     := $(wildcard *.cpp)
OBJS := $(patsubst %.cpp,%.o,$(SRC))

LIBRARY = libsycl.a

all: $(LIBRARY)

$(LIBRARY): $(OBJS)
	ar rcs $(LIBRARY) $(OBJS)

%.o: %.cpp
	$(DPCPP) $(DPCPPFLAGS) -I../../include -c $<

clean:
	rm -rf *.o

