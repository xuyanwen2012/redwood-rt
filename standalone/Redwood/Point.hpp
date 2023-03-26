#pragma once

#include <iostream>
#include <type_traits>

// The purpose of this File is just to provide a uniformed vector type for both
// client and backend code. In CUDA, there is a almost equivalent class called
// 'vector_types.h'. But if the client uses something different, then they have
// to convert that type to a backend type.
template <int Dim, typename T>
struct Point {
  static_assert(std::is_arithmetic<T>::value, "T must be numeric");
  static_assert(Dim > 0, "Dim must be greater than 0");

  Point() = default;

  T data[Dim];

  Point<Dim, T> operator/(const T a) const {
    Point<Dim, T> result;
    for (int i = 0; i < Dim; ++i) {
      result.data[i] = data[i] / a;
    }
    return result;
  }

  Point<Dim, T> operator*(const T a) const {
    Point<Dim, T> result;
    for (int i = 0; i < Dim; ++i) {
      result.data[i] = data[i] * a;
    }
    return result;
  }

  Point<Dim, T> operator+(const Point<Dim, T>& pos) const {
    Point<Dim, T> result;
    for (int i = 0; i < Dim; ++i) {
      result.data[i] = data[i] + pos.data[i];
    }
    return result;
  }

  Point<Dim, T> operator-(const Point<Dim, T>& pos) const {
    Point<Dim, T> result;
    for (int i = 0; i < Dim; ++i) {
      result.data[i] = data[i] - pos.data[i];
    }
    return result;
  }

  Point<Dim, T>& operator+=(const Point<Dim, T>& rhs) {
    for (int i = 0; i < Dim; ++i) {
      this->data[i] += rhs.data[i];
    }
    return *this;
  }

  Point<Dim, T>& operator++() {
    ++data[0];
    return *this;
  };
};

template <int Dim, typename T>
bool operator==(const Point<Dim, T>& a, const Point<Dim, T>& b) {
  for (int i = 0; i < Dim; ++i) {
    if (a.data[i] != b.data[i]) return false;
  }
  return true;
}

template <int Dim, typename T>
bool operator!=(const Point<Dim, T>& a, const Point<Dim, T>& b) {
  return !(a == b);
}

template <int Dim, typename T>
std::ostream& operator<<(std::ostream& os, const Point<Dim, T>& dt) {
  os << '(';
  for (int i = 0; i < Dim - 1; ++i) {
    os << dt.data[i] << ", ";
  }
  os << dt.data[Dim - 1] << ')';
  return os;
}

using Point2F = Point<2, float>;
using Point2D = Point<2, double>;
using Point3F = Point<3, float>;
using Point3D = Point<3, double>;
using Point4F = Point<4, float>;
using Point4D = Point<4, double>;