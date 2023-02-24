include ../../Makefile.inc

DUET_LIB := ../../accelerator/duet/libduet.a

SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

all: duet

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $(SOURCES) -I../../include

duet: $(OBJECTS)
	$(CXX) -o  duet.out $(OBJECTS) $(DUET_LIB)

clean:
	rm -f $(OBJECTS) *.out