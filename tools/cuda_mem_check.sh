#!/bin/sh
cuda-memcheck --leak-check full ./example/test/cuda.out
