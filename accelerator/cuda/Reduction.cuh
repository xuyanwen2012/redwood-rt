#pragma once

template <class T>
void reduce(int size, int threads, int blocks, int whichKernel, T *d_idata,
            T *d_odata);


const Point4F* lnt, const Point4F* u_q,
                                 const int* u_node_idx, float* u_out,
                                 const int num_active, const int max_leaf_size,
                                 const Functor functor