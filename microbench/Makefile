include ../Makefile.inc

SOURCES = $(wildcard *.cpp)

all: $(SOURCES:.cpp=.out)

# Compile each .cpp file separately into an executable
%.out: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(SRCS:.cpp=.out)
