CXX = g++

CXXFLAGS += -Wextra -Wall -Wfloat-equal -Wundef -Wshadow -Wpointer-arith -Wcast-align -Wstrict-prototypes -Wstrict-overflow=5 -Wwrite-strings -Waggregate-return -Wcast-qual -Wswitch-default -Wswitch-enum -Wconversion -Wunreachable-code -Wunused-result

CXXFLAGS += -std=c++17 -O3 -DNDEBUG

SYCL_LIB_DIR := /home/tsorensen/sycl_workspace/llvm/build/lib
