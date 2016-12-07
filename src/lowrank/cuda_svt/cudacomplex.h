/* Below is copied from fbcuda, BSD licensed */
#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>

/**
   `cuComplex` wrapper.
*/
struct singlecomplex {
  __host__ __device__ __forceinline__
  singlecomplex() {}

  __host__ __device__ __forceinline__
  singlecomplex(float re) : cplx_(make_cuComplex(re, 0.0f)) {}

  __host__ __device__ __forceinline__
  singlecomplex(float re, float im) : cplx_(make_cuComplex(re, im)) {}

  __host__ __device__ __forceinline__
  singlecomplex(const singlecomplex& c) : cplx_(c.cplx_) {}

  __host__ __device__ __forceinline__
  singlecomplex(const cuComplex& c) : cplx_(c) {}

  __host__ __device__ __forceinline__
  singlecomplex& operator=(const singlecomplex& c) {
    // No need for swap
    cplx_ = c.cplx_;
    return *this;
  }

  __host__ __device__ __forceinline__
  bool operator==(const singlecomplex& c) const {
    return cplx_.x == c.cplx_.x && cplx_.y == c.cplx_.y;
  }

  __host__ __device__ __forceinline__
  bool operator!=(const singlecomplex& c) const {
    return !operator==(c);
  }

  __host__ __device__ __forceinline__
  singlecomplex operator-() const {
    return singlecomplex(make_cuComplex(-cplx_.x, -cplx_.y));
  }

  __host__ __device__ __forceinline__
  singlecomplex operator-(const singlecomplex& c) const {
    return singlecomplex(cuCsubf(cplx_, c.cplx_));
  }

  __host__ __device__ __forceinline__
  singlecomplex operator+(const singlecomplex& c) const {
    return singlecomplex(cuCaddf(cplx_, c.cplx_));
  }

  __host__ __device__ __forceinline__
  singlecomplex operator*(const singlecomplex& c) const {
    return singlecomplex(cuCmulf(cplx_, c.cplx_));
  }

  __host__ __device__ __forceinline__
  singlecomplex operator/(const singlecomplex& c) const {
    return singlecomplex(cuCdivf(cplx_, c.cplx_));
  }

  __host__ __device__ __forceinline__
  singlecomplex& operator+=(const singlecomplex& c) {
    cplx_ = cuCaddf(cplx_, c.cplx_);
    return *this;
  }

  __host__ __device__ __forceinline__
  singlecomplex& operator-=(const singlecomplex& c) {
    cplx_ = cuCsubf(cplx_, c.cplx_);
    return *this;
  }

  __host__ __device__ __forceinline__
  singlecomplex& operator*=(const singlecomplex& c) {
    cplx_ = cuCmulf(cplx_, c.cplx_);
    return *this;
  }

  __host__ __device__ __forceinline__
  singlecomplex& operator/=(const singlecomplex& c) {
    cplx_ = cuCdivf(cplx_, c.cplx_);
    return *this;
  }

  __host__ __device__ __forceinline__
  singlecomplex transpose() const {
    return singlecomplex(make_cuComplex(cplx_.y, cplx_.x));
  }

  __host__ __device__ __forceinline__
  singlecomplex conjugate() const {
    return singlecomplex(make_cuComplex(cplx_.x, -cplx_.y));
  }

  __host__ __device__ __forceinline__
  void cexp(float angle) {
    sincosf(angle, &cplx_.y, &cplx_.x);
  }

  __host__ __device__ __forceinline__
  float& re() {
    return cplx_.x;
  }

  __host__ __device__ __forceinline__
  float& im() {
    return cplx_.y;
  }

  __host__ __device__ __forceinline__
  const float& re() const {
    return cplx_.x;
  }

  __host__ __device__ __forceinline__
  const float& im() const {
    return cplx_.y;
  }

  __host__ __device__ __forceinline__
  float abs() {
    return cuCabsf(cplx_);
  }

  __host__ __device__ __forceinline__
  float norm() {
    return cplx_.x * cplx_.x + cplx_.y * cplx_.y;
  }

  __host__ __device__ __forceinline__
  operator float2() const {
    return static_cast<float2>(cplx_);
  }

private:
  cuComplex cplx_;
};
