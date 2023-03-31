DPCPP = clang++ -fsycl
DPCPPFLAGS += -std=c++17 -Wall -Wunused-result -O3 -DNDEBUG

SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

LIBRARY = libredwoodsycl.a

all: $(LIBRARY)

%.o: %.cpp
	$(DPCPP) $(DPCPPFLAGS) -c $< -o $@ -I../../include

$(LIBRARY): $(OBJECTS)
	ar rcs $@ $^

clean:
	rm -r *.o $(LIBRARY)

