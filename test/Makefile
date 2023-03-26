G_BENCH_INCLUDE := -isystem ~/benchmark/include -L ~/benchmark/build/src/ -lbenchmark
G_TEST_INCLUDE := -I ~/googletest/googletest/include/ -L ~/googletest/build/lib/ -lgtest

all:
	g++ Main2.cpp --std=c++17 $(G_TEST_INCLUDE) -lgtest_main -lpthread
