#pragma once

#include "Point.hpp"

namespace redwood {

void LaunchNnKenrnel(const int* u_leaf_indices, /**/
                     const Point4F* u_q_points, /**/
                     int num_active_leafs,      /**/
                     float* u_out,              /* stream base addr */
                     const Point4F* u_lnt_data, /**/
                     int max_leaf_size, int stream_id);

}