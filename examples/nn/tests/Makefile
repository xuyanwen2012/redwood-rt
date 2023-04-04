G_BENCH_INCLUDE := -isystem ~/benchmark/include -L ~/benchmark/build/src/ -lbenchmark
G_TEST_INCLUDE := -I ~/googletest/googletest/include/ -L ~/googletest/build/lib/ -lgtest

APP_INCLUDE := -I ../../../include/

all:
	g++ Query.cpp --std=c++17 $(APP_INCLUDE) $(G_TEST_INCLUDE) -lgtest_main -lpthread

clean:
	rm -f *.out
