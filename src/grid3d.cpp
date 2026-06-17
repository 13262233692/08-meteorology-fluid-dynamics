#include "grid3d.h"
#include <cstring>
#include <algorithm>

namespace mfd {

Grid3D::Grid3D() {
    dims_.nx = 0;
    dims_.ny = 0;
    dims_.nz = 0;
    dims_.dx = 1.0;
    dims_.dy = 1.0;
    dims_.dz = 1.0;
    dims_.origin_x = 0.0;
    dims_.origin_y = 0.0;
    dims_.origin_z = 0.0;
}

Grid3D::Grid3D(size_t nx, size_t ny, size_t nz,
                double dx, double dy, double dz,
                double ox, double oy, double oz) {
    dims_.nx = nx;
    dims_.ny = ny;
    dims_.nz = nz;
    dims_.dx = dx;
    dims_.dy = dy;
    dims_.dz = dz;
    dims_.origin_x = ox;
    dims_.origin_y = oy;
    dims_.origin_z = oz;
    allocate(nx, ny, nz);
}

Grid3D::~Grid3D() = default;

Grid3D::Grid3D(const Grid3D& other)
    : dims_(other.dims_), data_(other.data_) {}

Grid3D::Grid3D(Grid3D&& other) noexcept
    : dims_(other.dims_), data_(std::move(other.data_)) {
    other.dims_.nx = 0;
    other.dims_.ny = 0;
    other.dims_.nz = 0;
}

Grid3D& Grid3D::operator=(const Grid3D& other) {
    if (this != &other) {
        dims_ = other.dims_;
        data_ = other.data_;
    }
    return *this;
}

Grid3D& Grid3D::operator=(Grid3D&& other) noexcept {
    if (this != &other) {
        dims_ = other.dims_;
        data_ = std::move(other.data_);
        other.dims_.nx = 0;
        other.dims_.ny = 0;
        other.dims_.nz = 0;
    }
    return *this;
}

void Grid3D::allocate(size_t nx, size_t ny, size_t nz) {
    dims_.nx = nx;
    dims_.ny = ny;
    dims_.nz = nz;
    data_.assign(nx * ny * nz, 0.0);
}

void Grid3D::clear() {
    dims_.nx = 0;
    dims_.ny = 0;
    dims_.nz = 0;
    data_.clear();
}

void Grid3D::fill(double value) {
    std::fill(data_.begin(), data_.end(), value);
}

void Grid3D::copy_from(const double* src, size_t count) {
    size_t n = std::min(count, data_.size());
    std::memcpy(data_.data(), src, n * sizeof(double));
}

VectorField3D::VectorField3D() = default;

VectorField3D::VectorField3D(size_t nx, size_t ny, size_t nz,
                           double dx, double dy, double dz)
    : u(nx, ny, nz, dx, dy, dz),
      v(nx, ny, nz, dx, dy, dz),
      w(nx, ny, nz, dx, dy, dz) {}

Grid3D& VectorField3D::component(int idx) {
    switch (idx) {
        case 0: return u;
        case 1: return v;
        case 2: return w;
        default: return u;
    }
}

const Grid3D& VectorField3D::component(int idx) const {
    switch (idx) {
        case 0: return u;
        case 1: return v;
        case 2: return w;
        default: return u;
    }
}

void VectorField3D::allocate(size_t nx, size_t ny, size_t nz) {
    u.allocate(nx, ny, nz);
    v.allocate(nx, ny, nz);
    w.allocate(nx, ny, nz);
}

}  // namespace mfd
