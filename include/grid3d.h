#ifndef MFD_GRID3D_H
#define MFD_GRID3D_H

#include <vector>
#include <string>
#include <cstddef>
#include <memory>

namespace mfd {

struct GridDimensions {
    size_t nx;
    size_t ny;
    size_t nz;
    double dx;
    double dy;
    double dz;
    double origin_x;
    double origin_y;
    double origin_z;
};

class Grid3D {
public:
    Grid3D();
    Grid3D(size_t nx, size_t ny, size_t nz,
            double dx = 1.0, double dy = 1.0, double dz = 1.0,
            double ox = 0.0, double oy = 0.0, double oz = 0.0);
    ~Grid3D();

    Grid3D(const Grid3D& other);
    Grid3D(Grid3D&& other) noexcept;
    Grid3D& operator=(const Grid3D& other);
    Grid3D& operator=(Grid3D&& other) noexcept;

    void allocate(size_t nx, size_t ny, size_t nz);
    void clear();

    inline size_t nx() const { return dims_.nx; }
    inline size_t ny() const { return dims_.ny; }
    inline size_t nz() const { return dims_.nz; }
    inline size_t size() const { return dims_.nx * dims_.ny * dims_.nz; }

    inline const GridDimensions& dimensions() const { return dims_; }
    inline GridDimensions& dimensions() { return dims_; }

    inline double& operator()(size_t i, size_t j, size_t k) {
        return data_[index(i, j, k)];
    }

    inline const double& operator()(size_t i, size_t j, size_t k) const {
        return data_[index(i, j, k)];
    }

    inline double* data() { return data_.data(); }
    inline const double* data() const { return data_.data(); }

    inline std::vector<double>& data_vector() { return data_; }
    inline const std::vector<double>& data_vector() const { return data_; }

    void fill(double value);
    void copy_from(const double* src, size_t count);

    inline size_t index(size_t i, size_t j, size_t k) const {
        return i + dims_.nx * (j + dims_.ny * k);
    }

private:
    GridDimensions dims_;
    std::vector<double> data_;
};

class VectorField3D {
public:
    VectorField3D();
    VectorField3D(size_t nx, size_t ny, size_t nz,
                  double dx = 1.0, double dy = 1.0, double dz = 1.0);

    Grid3D u;
    Grid3D v;
    Grid3D w;

    Grid3D& component(int idx);
    const Grid3D& component(int idx) const;

    size_t nx() const { return u.nx(); }
    size_t ny() const { return u.ny(); }
    size_t nz() const { return u.nz(); }

    void allocate(size_t nx, size_t ny, size_t nz);
};

}  // namespace mfd

#endif  // MFD_GRID3D_H
