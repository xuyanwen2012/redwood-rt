include ../../Makefile.inc

# SRC     := $(wildcard *.cpp)
# OBJS := $(patsubst %.cpp,%.o,$(SRC))

LIBRARY = libduet.a

# all: $(LIBRARY)

all: 
	$(CXX) $(CXXFLAGS) -I../../include -c Core.cpp Usm.cpp ./barnes/BarnesReducer.cpp

$(LIBRARY):
	ar rcs $(LIBRARY) Core.o Usm.o BarnesReducer.o

# %.o: %.cpp
# 	$(CXX) $(CXXFLAGS) -I../../include -c $<

clean:
	rm -rf *.o $(LIBRARY)

