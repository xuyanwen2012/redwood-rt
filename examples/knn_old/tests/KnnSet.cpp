
#include <algorithm>
#include <iostream>
#include <limits>

// template <typename T, int K>
// class MinHeap {
//  public:
//   MinHeap() { std::fill_n(heap_, K, std::numeric_limits<T>::max()); }

//   void Insert(T value) {
//     if (value >= heap_[0]) {
//       // Value is too large, discard it
//       return;
//     }
//     heap_[0] = value;
//     Heapify(0);
//   }

//   void DebugPrint() const {
//     for (int i = 0; i < K; ++i) {
//       std::cout << i << ":\t" << heap_[i] << '\n';
//     }
//     std::cout << std::endl;
//   }

//   const T* Data() const { return heap_; }

//  private:
//   T heap_[K];

//   void Heapify(int i) {
//     int left = 2 * i + 1;
//     int right = 2 * i + 2;
//     int smallest = i;

//     if (left < K && heap_[left] < heap_[smallest]) {
//       smallest = left;
//     }
//     if (right < K && heap_[right] < heap_[smallest]) {
//       smallest = right;
//     }

//     if (smallest != i) {
//       std::swap(heap_[i], heap_[smallest]);
//       Heapify(smallest);
//     }
//   }
// };

int main() {
  PriorityQueue queue;

  queue.DebugPrint();

  for (int i = 64; i >= 0; --i) {
    queue.push(i);
  }
  queue.DebugPrint();

  std::cout << queue.top() << std::endl;
  queue.pop();
  std::cout << queue.top() << std::endl;
  queue.pop();

  std::cout << queue.top() << std::endl;

  return 0;
}
