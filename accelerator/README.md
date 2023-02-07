# Accelerator Backends

## APIs

There are some APIs must be implemented for each new backends. 

* Everything in `include/accelerator/Core.hpp`
* Everything in `include/accelerator/Usm.hpp`
* Something in `include/accelerator/Kernels.hpp`

## Backends

### CPU

Note the CPU backend is only for testing application and Debugging library, it is not an actual CPU accelerator. 

### CUDA

For NVIDIA GPUs, in particular, the **Jetson** Series. 

### SYCL 

For Intel GPUs

### Duet

The Duet FPGA, must run with Gem5 Simulator
