#pragma once

namespace redwood::rt {

// Developer APIs
// Redwood developers can use the following APIs to micro controll the execution
// details. This particular function is used for GPU backend Executor Runtime.
void ExecuteCurrentBufferAsync(int tid, int num_batch_collected);

void ExecuteBuffer(int tid, int stream_id, int num_batch_collected);

}  // namespace redwood::rt