# Redwood heterogeneous framework

## Redwood and Grove

Working on Documentation Now, instructions coming soon

Old SYCL backend was in https://github.com/xuyanwen2012/new-redwood

Old CUDA backend was in https://github.com/xuyanwen2012/redwood

## Getting Started

### Compile Backends

To start, the example accelerator code (e.g., CUDA, SYCL) are located in the `accelerator/cuda` folder.

```
cd ./accelerator/cuda/
make
```

It will build a static library. 'nvcc' is required. 

### Compile Applications

Example application code (e.g., nearest neighbor, barnus-hut) are located in the `examples` folder. And sample input data are located in 'data' folder. 

```
cd ./examples/nn/
make cuda
```

It will build a nn application using the CUDA backend. To run the application, `./cuda`

```
requires an input file ("data/input_nn_1m_4f.dat")
Redwood NN demo implementation
Usage:
  Nearest Neighbor (NN) [OPTION...] positional parameters

  -t, --thread arg      Number of threads (default: 1)
  -l, --leaf arg        Maximum leaf node size (default: 32)
  -b, --batch_size arg  Batch size (GPU) (default: 1024)
  -c, --cpu             Enable CPU baseline
  -h, --help            Print usage
```

Here are some example input arguments

```
// Running CPU baseline using `-c` flag
./cuda ../../data/1m_nn_uniform_nn_4f.dat -c

// Running GPU with large 
./cuda ../../data/1m_nn_uniform_nn_4f.dat -l 1024

```

## Gem5 Simulator Code

https://github.com/angl-dev/gem5-duet/tree/Yanwen

### Misc

`/usr/local/llvm-14.0/bin/clang-format`

## Some workloads lives in another repository




