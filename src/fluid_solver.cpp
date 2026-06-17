#include "fluid_solver.h"
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>

namespace mfd {

FluidSolver::FluidSolver(const SolverConfig& config)
    : config_(config) {
    if (config_.num_threads > 0) {
        omp_set_num_threads(config_.num_threads);
    }
}

FluidSolver::~FluidSolver() = default;

inline bool FluidSolver::is_valid(double val) const {
    return std::isfinite(val);
}

inline double FluidSolver::safe_val(double val, double fallback) const {
    return std::isfinite(val) ? val : fallback;
}

void FluidSolver::sanitize_grid(Grid3D& grid, double fill_value) const {
    const int nx = static_cast<int>(grid.nx());
    const int ny = static_cast<int>(grid.ny());
    const int nz = static_cast<int>(grid.nz());

    #pragma omp parallel for schedule(static)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                double val = grid(i, j, k);
                if (!std::isfinite(val)) {
                    grid(i, j, k) = fill_value;
                }
            }
        }
    }
}

void FluidSolver::sanitize_vector_field(VectorField3D& vf, double fill_value) const {
    sanitize_grid(vf.u, fill_value);
    sanitize_grid(vf.v, fill_value);
    sanitize_grid(vf.w, fill_value);
}

size_t FluidSolver::count_nan(const Grid3D& grid) const {
    const int nx = static_cast<int>(grid.nx());
    const int ny = static_cast<int>(grid.ny());
    const int nz = static_cast<int>(grid.nz());
    size_t count = 0;

    #pragma omp parallel for reduction(+:count) schedule(static)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                if (!std::isfinite(grid(i, j, k))) {
                    ++count;
                }
            }
        }
    }
    return count;
}

inline double FluidSolver::ddx_central(const Grid3D& f, size_t i, size_t j, size_t k) const {
    size_t ip = std::min(i + 1, f.nx() - 1);
    size_t im = (i > 0) ? (i - 1) : 0;
    double fp = safe_val(f(ip, j, k), 0.0);
    double fm = safe_val(f(im, j, k), 0.0);
    return (fp - fm) / (2.0 * f.dimensions().dx);
}

inline double FluidSolver::ddy_central(const Grid3D& f, size_t i, size_t j, size_t k) const {
    size_t jp = std::min(j + 1, f.ny() - 1);
    size_t jm = (j > 0) ? (j - 1) : 0;
    double fp = safe_val(f(i, jp, k), 0.0);
    double fm = safe_val(f(i, jm, k), 0.0);
    return (fp - fm) / (2.0 * f.dimensions().dy);
}

inline double FluidSolver::ddz_central(const Grid3D& f, size_t i, size_t j, size_t k) const {
    size_t kp = std::min(k + 1, f.nz() - 1);
    size_t km = (k > 0) ? (k - 1) : 0;
    double fp = safe_val(f(i, j, kp), 0.0);
    double fm = safe_val(f(i, j, km), 0.0);
    return (fp - fm) / (2.0 * f.dimensions().dz);
}

inline double FluidSolver::laplacian(const Grid3D& f, size_t i, size_t j, size_t k) const {
    size_t ip = std::min(i + 1, f.nx() - 1);
    size_t im = (i > 0) ? (i - 1) : 0;
    size_t jp = std::min(j + 1, f.ny() - 1);
    size_t jm = (j > 0) ? (j - 1) : 0;
    size_t kp = std::min(k + 1, f.nz() - 1);
    size_t km = (k > 0) ? (k - 1) : 0;

    double fc = safe_val(f(i, j, k), 0.0);
    double f_ip = safe_val(f(ip, j, k), 0.0);
    double f_im = safe_val(f(im, j, k), 0.0);
    double f_jp = safe_val(f(i, jp, k), 0.0);
    double f_jm = safe_val(f(i, jm, k), 0.0);
    double f_kp = safe_val(f(i, j, kp), 0.0);
    double f_km = safe_val(f(i, j, km), 0.0);

    double dx2 = f.dimensions().dx * f.dimensions().dx;
    double dy2 = f.dimensions().dy * f.dimensions().dy;
    double dz2 = f.dimensions().dz * f.dimensions().dz;

    return (f_ip - 2.0 * fc + f_im) / dx2 +
           (f_jp - 2.0 * fc + f_jm) / dy2 +
           (f_kp - 2.0 * fc + f_km) / dz2;
}

void FluidSolver::compute_divergence(const VectorField3D& velocity, Grid3D& divergence) const {
    const int nx = static_cast<int>(velocity.nx());
    const int ny = static_cast<int>(velocity.ny());
    const int nz = static_cast<int>(velocity.nz());

    if (divergence.nx() != static_cast<size_t>(nx) ||
        divergence.ny() != static_cast<size_t>(ny) ||
        divergence.nz() != static_cast<size_t>(nz)) {
        divergence.allocate(static_cast<size_t>(nx), static_cast<size_t>(ny), static_cast<size_t>(nz));
        divergence.dimensions() = velocity.u.dimensions();
    }
    divergence.fill(0.0);

    #pragma omp parallel for schedule(static, 1)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                double dudx = ddx_central(velocity.u, i, j, k);
                double dvdy = ddy_central(velocity.v, i, j, k);
                double dwdz = ddz_central(velocity.w, i, j, k);
                double div = dudx + dvdy + dwdz;
                divergence(i, j, k) = safe_val(div, 0.0);
            }
        }
    }
}

void FluidSolver::compute_vorticity(const VectorField3D& velocity, VectorField3D& vorticity) const {
    const int nx = static_cast<int>(velocity.nx());
    const int ny = static_cast<int>(velocity.ny());
    const int nz = static_cast<int>(velocity.nz());

    if (vorticity.nx() != static_cast<size_t>(nx) ||
        vorticity.ny() != static_cast<size_t>(ny) ||
        vorticity.nz() != static_cast<size_t>(nz)) {
        vorticity.allocate(static_cast<size_t>(nx), static_cast<size_t>(ny), static_cast<size_t>(nz));
        vorticity.u.dimensions() = velocity.u.dimensions();
        vorticity.v.dimensions() = velocity.u.dimensions();
        vorticity.w.dimensions() = velocity.u.dimensions();
    }
    vorticity.u.fill(0.0);
    vorticity.v.fill(0.0);
    vorticity.w.fill(0.0);

    #pragma omp parallel for schedule(static, 1)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                double dwdy = ddy_central(velocity.w, i, j, k);
                double dvdz = ddz_central(velocity.v, i, j, k);
                vorticity.u(i, j, k) = safe_val(dwdy - dvdz, 0.0);

                double dudz = ddz_central(velocity.u, i, j, k);
                double dwdx = ddx_central(velocity.w, i, j, k);
                vorticity.v(i, j, k) = safe_val(dudz - dwdx, 0.0);

                double dvdx = ddx_central(velocity.v, i, j, k);
                double dudy = ddy_central(velocity.u, i, j, k);
                vorticity.w(i, j, k) = safe_val(dvdx - dudy, 0.0);
            }
        }
    }
}

void FluidSolver::compute_vorticity_magnitude(const VectorField3D& vorticity, Grid3D& magnitude) const {
    const int nx = static_cast<int>(vorticity.nx());
    const int ny = static_cast<int>(vorticity.ny());
    const int nz = static_cast<int>(vorticity.nz());

    if (magnitude.nx() != static_cast<size_t>(nx) ||
        magnitude.ny() != static_cast<size_t>(ny) ||
        magnitude.nz() != static_cast<size_t>(nz)) {
        magnitude.allocate(static_cast<size_t>(nx), static_cast<size_t>(ny), static_cast<size_t>(nz));
        magnitude.dimensions() = vorticity.u.dimensions();
    }
    magnitude.fill(0.0);

    #pragma omp parallel for schedule(static, 1)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                double wx = safe_val(vorticity.u(i, j, k), 0.0);
                double wy = safe_val(vorticity.v(i, j, k), 0.0);
                double wz = safe_val(vorticity.w(i, j, k), 0.0);
                double mag = std::sqrt(wx * wx + wy * wy + wz * wz);
                magnitude(i, j, k) = safe_val(mag, 0.0);
            }
        }
    }
}

void FluidSolver::apply_boundary_conditions(VectorField3D& velocity) const {
    const int nx = static_cast<int>(velocity.nx());
    const int ny = static_cast<int>(velocity.ny());
    const int nz = static_cast<int>(velocity.nz());

    if (nx < 2 || ny < 2 || nz < 2) return;

    #pragma omp parallel for schedule(static, 1)
    for (int idx = 0; idx < (2 * ny * nz + 2 * (nx - 2) * nz + 2 * (nx - 2) * (ny - 2)); ++idx) {
        int i, j, k;
        int face = 0;
        int rem = idx;

        if (rem < 2 * ny * nz) {
            face = (rem < ny * nz) ? 0 : 1;
            rem = rem % (ny * nz);
            j = rem / nz;
            k = rem % nz;
            i = (face == 0) ? 0 : (nx - 1);
        } else {
            rem -= 2 * ny * nz;
            if (rem < 2 * (nx - 2) * nz) {
                face = (rem < (nx - 2) * nz) ? 2 : 3;
                rem = rem % ((nx - 2) * nz);
                i = (rem / nz) + 1;
                k = rem % nz;
                j = (face == 2) ? 0 : (ny - 1);
            } else {
                rem -= 2 * (nx - 2) * nz;
                face = (rem < (nx - 2) * (ny - 2)) ? 4 : 5;
                rem = rem % ((nx - 2) * (ny - 2));
                i = (rem / (ny - 2)) + 1;
                j = (rem % (ny - 2)) + 1;
                k = (face == 4) ? 0 : (nz - 1);
            }
        }

        velocity.u(i, j, k) = 0.0;
        velocity.v(i, j, k) = 0.0;
        velocity.w(i, j, k) = 0.0;
    }
}

void FluidSolver::diffuse_velocity(VectorField3D& velocity, double dt) const {
    const int nx = static_cast<int>(velocity.nx());
    const int ny = static_cast<int>(velocity.ny());
    const int nz = static_cast<int>(velocity.nz());
    const double nu = config_.kinematic_viscosity;

    VectorField3D temp(static_cast<size_t>(nx), static_cast<size_t>(ny), static_cast<size_t>(nz));
    temp.u.dimensions() = velocity.u.dimensions();
    temp.v.dimensions() = velocity.u.dimensions();
    temp.w.dimensions() = velocity.u.dimensions();

    #pragma omp parallel for schedule(static, 1)
    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                double lap_u = laplacian(velocity.u, i, j, k);
                double lap_v = laplacian(velocity.v, i, j, k);
                double lap_w = laplacian(velocity.w, i, j, k);

                double u_new = safe_val(velocity.u(i, j, k), 0.0) + nu * dt * lap_u;
                double v_new = safe_val(velocity.v(i, j, k), 0.0) + nu * dt * lap_v;
                double w_new = safe_val(velocity.w(i, j, k), 0.0) + nu * dt * lap_w;

                temp.u(i, j, k) = safe_val(u_new, 0.0);
                temp.v(i, j, k) = safe_val(v_new, 0.0);
                temp.w(i, j, k) = safe_val(w_new, 0.0);
            }
        }
    }

    velocity.u.data_vector().swap(temp.u.data_vector());
    velocity.v.data_vector().swap(temp.v.data_vector());
    velocity.w.data_vector().swap(temp.w.data_vector());

    apply_boundary_conditions(velocity);
}

void FluidSolver::advect_velocity(VectorField3D& velocity, double dt) const {
    const int nx = static_cast<int>(velocity.nx());
    const int ny = static_cast<int>(velocity.ny());
    const int nz = static_cast<int>(velocity.nz());
    const auto& dims = velocity.u.dimensions();

    VectorField3D temp(static_cast<size_t>(nx), static_cast<size_t>(ny), static_cast<size_t>(nz));
    temp.u.dimensions() = dims;
    temp.v.dimensions() = dims;
    temp.w.dimensions() = dims;

    const double max_displacement = static_cast<double>(std::min({nx, ny, nz})) * 0.45;

    #pragma omp parallel for schedule(static, 1)
    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                double u = safe_val(velocity.u(i, j, k), 0.0);
                double v = safe_val(velocity.v(i, j, k), 0.0);
                double w = safe_val(velocity.w(i, j, k), 0.0);

                double dx = u * dt / dims.dx;
                double dy = v * dt / dims.dy;
                double dz = w * dt / dims.dz;

                double disp = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (disp > max_displacement) {
                    double scale = max_displacement / disp;
                    dx *= scale;
                    dy *= scale;
                    dz *= scale;
                }

                double x = static_cast<double>(i) - dx;
                double y = static_cast<double>(j) - dy;
                double z = static_cast<double>(k) - dz;

                x = std::max(0.5, std::min(static_cast<double>(nx - 1) - 0.5, x));
                y = std::max(0.5, std::min(static_cast<double>(ny - 1) - 0.5, y));
                z = std::max(0.5, std::min(static_cast<double>(nz - 1) - 0.5, z));

                if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
                    temp.u(i, j, k) = u;
                    temp.v(i, j, k) = v;
                    temp.w(i, j, k) = w;
                    continue;
                }

                int ix = static_cast<int>(std::floor(x));
                int iy = static_cast<int>(std::floor(y));
                int iz = static_cast<int>(std::floor(z));

                ix = std::max(0, std::min(nx - 2, ix));
                iy = std::max(0, std::min(ny - 2, iy));
                iz = std::max(0, std::min(nz - 2, iz));

                int ix1 = std::min(ix + 1, nx - 1);
                int iy1 = std::min(iy + 1, ny - 1);
                int iz1 = std::min(iz + 1, nz - 1);

                double tx = std::max(0.0, std::min(1.0, x - ix));
                double ty = std::max(0.0, std::min(1.0, y - iy));
                double tz = std::max(0.0, std::min(1.0, z - iz));

                auto interp = [&](const Grid3D& f) -> double {
                    double f000 = safe_val(f(ix, iy, iz), 0.0);
                    double f100 = safe_val(f(ix1, iy, iz), 0.0);
                    double f010 = safe_val(f(ix, iy1, iz), 0.0);
                    double f110 = safe_val(f(ix1, iy1, iz), 0.0);
                    double f001 = safe_val(f(ix, iy, iz1), 0.0);
                    double f101 = safe_val(f(ix1, iy, iz1), 0.0);
                    double f011 = safe_val(f(ix, iy1, iz1), 0.0);
                    double f111 = safe_val(f(ix1, iy1, iz1), 0.0);

                    double c00 = f000 * (1 - tx) + f100 * tx;
                    double c10 = f010 * (1 - tx) + f110 * tx;
                    double c01 = f001 * (1 - tx) + f101 * tx;
                    double c11 = f011 * (1 - tx) + f111 * tx;

                    double c0 = c00 * (1 - ty) + c10 * ty;
                    double c1 = c01 * (1 - ty) + c11 * ty;

                    return c0 * (1 - tz) + c1 * tz;
                };

                double new_u = interp(velocity.u);
                double new_v = interp(velocity.v);
                double new_w = interp(velocity.w);

                temp.u(i, j, k) = safe_val(new_u, u);
                temp.v(i, j, k) = safe_val(new_v, v);
                temp.w(i, j, k) = safe_val(new_w, w);
            }
        }
    }

    velocity.u.data_vector().swap(temp.u.data_vector());
    velocity.v.data_vector().swap(temp.v.data_vector());
    velocity.w.data_vector().swap(temp.w.data_vector());

    apply_boundary_conditions(velocity);
}

void FluidSolver::add_buoyancy_term(VectorField3D& velocity, const Grid3D& reflectivity) const {
    const int nx = static_cast<int>(velocity.nx());
    const int ny = static_cast<int>(velocity.ny());
    const int nz = static_cast<int>(velocity.nz());
    const double g = 9.81;
    const double rho0 = config_.density;

    #pragma omp parallel for schedule(static, 1)
    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                double dbz = safe_val(reflectivity(i, j, k), 0.0);
                if (dbz > 10.0) {
                    double buoyancy = g * (dbz - 10.0) / (50.0 * rho0);
                    double w_new = safe_val(velocity.w(i, j, k), 0.0) + buoyancy * config_.time_step;
                    velocity.w(i, j, k) = safe_val(w_new, 0.0);
                }
            }
        }
    }
}

void FluidSolver::compute_pressure_poisson(const VectorField3D& velocity, Grid3D& pressure) const {
    const int nx = static_cast<int>(velocity.nx());
    const int ny = static_cast<int>(velocity.ny());
    const int nz = static_cast<int>(velocity.nz());
    const auto& dims = velocity.u.dimensions();

    Grid3D rhs(static_cast<size_t>(nx), static_cast<size_t>(ny), static_cast<size_t>(nz),
                dims.dx, dims.dy, dims.dz);

    #pragma omp parallel for schedule(static, 1)
    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                double dudx = ddx_central(velocity.u, i, j, k);
                double dvdy = ddy_central(velocity.v, i, j, k);
                double dwdz = ddz_central(velocity.w, i, j, k);
                double div = dudx + dvdy + dwdz;
                rhs(i, j, k) = safe_val(div, 0.0) / config_.time_step;
            }
        }
    }

    pressure.fill(0.0);
    Grid3D pressure_new(static_cast<size_t>(nx), static_cast<size_t>(ny), static_cast<size_t>(nz),
                         dims.dx, dims.dy, dims.dz);
    pressure_new.fill(0.0);

    double dx2 = dims.dx * dims.dx;
    double dy2 = dims.dy * dims.dy;
    double dz2 = dims.dz * dims.dz;
    double inv_coeff = 1.0 / (2.0 / dx2 + 2.0 / dy2 + 2.0 / dz2);

    const int jacobi_iterations = 80;
    for (int iter = 0; iter < jacobi_iterations; ++iter) {
        #pragma omp parallel for schedule(static, 1)
        for (int k = 1; k < nz - 1; ++k) {
            for (int j = 1; j < ny - 1; ++j) {
                for (int i = 1; i < nx - 1; ++i) {
                    double p_ip = safe_val(pressure(i + 1, j, k), 0.0);
                    double p_im = safe_val(pressure(i - 1, j, k), 0.0);
                    double p_jp = safe_val(pressure(i, j + 1, k), 0.0);
                    double p_jm = safe_val(pressure(i, j - 1, k), 0.0);
                    double p_kp = safe_val(pressure(i, j, k + 1), 0.0);
                    double p_km = safe_val(pressure(i, j, k - 1), 0.0);

                    double sum = (p_ip + p_im) / dx2 +
                                 (p_jp + p_jm) / dy2 +
                                 (p_kp + p_km) / dz2;

                    double r = safe_val(rhs(i, j, k), 0.0);
                    double p_new = (sum - r) * inv_coeff;
                    pressure_new(i, j, k) = safe_val(p_new, 0.0);
                }
            }
        }

        #pragma omp barrier

        #pragma omp parallel for schedule(static, 1)
        for (int k = 1; k < nz - 1; ++k) {
            for (int j = 1; j < ny - 1; ++j) {
                for (int i = 1; i < nx - 1; ++i) {
                    pressure(i, j, k) = pressure_new(i, j, k);
                }
            }
        }

        #pragma omp barrier
    }
}

void FluidSolver::apply_pressure_gradient(VectorField3D& velocity, const Grid3D& pressure) const {
    const int nx = static_cast<int>(velocity.nx());
    const int ny = static_cast<int>(velocity.ny());
    const int nz = static_cast<int>(velocity.nz());
    const double rho = config_.density;
    const double dt_over_rho = config_.time_step / rho;

    #pragma omp parallel for schedule(static, 1)
    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                double dpdx = ddx_central(pressure, i, j, k);
                double dpdy = ddy_central(pressure, i, j, k);
                double dpdz = ddz_central(pressure, i, j, k);

                double u_new = safe_val(velocity.u(i, j, k), 0.0) - dt_over_rho * dpdx;
                double v_new = safe_val(velocity.v(i, j, k), 0.0) - dt_over_rho * dpdy;
                double w_new = safe_val(velocity.w(i, j, k), 0.0) - dt_over_rho * dpdz;

                velocity.u(i, j, k) = safe_val(u_new, 0.0);
                velocity.v(i, j, k) = safe_val(v_new, 0.0);
                velocity.w(i, j, k) = safe_val(w_new, 0.0);
            }
        }
    }
}

void FluidSolver::enforce_incompressibility(VectorField3D& velocity, int iterations) {
    Grid3D pressure;
    const auto& dims = velocity.u.dimensions();
    pressure.allocate(dims.nx, dims.ny, dims.nz);
    pressure.dimensions() = dims;
    pressure.fill(0.0);

    for (int iter = 0; iter < iterations; ++iter) {
        compute_pressure_poisson(velocity, pressure);
        apply_pressure_gradient(velocity, pressure);
    }

    apply_boundary_conditions(velocity);
}

void FluidSolver::compute_velocity_from_radial(const Grid3D& radial_wind,
                                                const Grid3D& reflectivity,
                                                VectorField3D& velocity,
                                                double elevation,
                                                double azimuth) {
    const int nx = static_cast<int>(radial_wind.nx());
    const int ny = static_cast<int>(radial_wind.ny());
    const int nz = static_cast<int>(radial_wind.nz());

    if (velocity.nx() != static_cast<size_t>(nx) ||
        velocity.ny() != static_cast<size_t>(ny) ||
        velocity.nz() != static_cast<size_t>(nz)) {
        velocity.allocate(static_cast<size_t>(nx), static_cast<size_t>(ny), static_cast<size_t>(nz));
    }
    velocity.u.dimensions() = radial_wind.dimensions();
    velocity.v.dimensions() = radial_wind.dimensions();
    velocity.w.dimensions() = radial_wind.dimensions();

    double cos_el = std::cos(elevation * M_PI / 180.0);
    double sin_el = std::sin(elevation * M_PI / 180.0);
    double cos_az = std::cos(azimuth * M_PI / 180.0);
    double sin_az = std::sin(azimuth * M_PI / 180.0);

    #pragma omp parallel for schedule(static, 1)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                double vr = radial_wind(i, j, k);
                if (!std::isfinite(vr)) {
                    velocity.u(i, j, k) = 0.0;
                    velocity.v(i, j, k) = 0.0;
                    velocity.w(i, j, k) = 0.0;
                    continue;
                }
                velocity.u(i, j, k) = vr * cos_el * cos_az;
                velocity.v(i, j, k) = vr * cos_el * sin_az;
                velocity.w(i, j, k) = vr * sin_el;
            }
        }
    }

    sanitize_vector_field(velocity, 0.0);
    apply_boundary_conditions(velocity);
    enforce_incompressibility(velocity, 30);
    (void)reflectivity;
}

void FluidSolver::solve_navier_stokes_simplified(VectorField3D& velocity,
                                                   const Grid3D& reflectivity,
                                                   int iterations) {
    int iters = (iterations > 0) ? iterations : config_.max_iterations;

    sanitize_vector_field(velocity, 0.0);
    apply_boundary_conditions(velocity);

    for (int iter = 0; iter < iters; ++iter) {
        size_t nan_count_before = count_nan(velocity.u) + count_nan(velocity.v) + count_nan(velocity.w);
        if (nan_count_before > 0) {
            std::cerr << "  [WARNING] NaN detected before NS iteration " << iter
                      << ": " << nan_count_before << " values" << std::endl;
            sanitize_vector_field(velocity, 0.0);
        }

        add_buoyancy_term(velocity, reflectivity);
        advect_velocity(velocity, config_.time_step);
        diffuse_velocity(velocity, config_.time_step);
        enforce_incompressibility(velocity, 20);
        apply_boundary_conditions(velocity);

        size_t nan_count_after = count_nan(velocity.u) + count_nan(velocity.v) + count_nan(velocity.w);
        if (nan_count_after > 0) {
            std::cerr << "  [WARNING] NaN detected after NS iteration " << iter
                      << ": " << nan_count_after << " values, sanitizing..." << std::endl;
            sanitize_vector_field(velocity, 0.0);
        }

        Grid3D div;
        compute_divergence(velocity, div);

        const int nx = static_cast<int>(div.nx());
        const int ny = static_cast<int>(div.ny());
        const int nz = static_cast<int>(div.nz());

        double max_div = 0.0;
        #pragma omp parallel for reduction(max:max_div) schedule(static)
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    double d = std::abs(safe_val(div(i, j, k), 0.0));
                    if (d > max_div) max_div = d;
                }
            }
        }

        if (iter % 10 == 0) {
            std::cout << "  NS iteration " << iter << ": max divergence = " << max_div << std::endl;
        }

        if (max_div < config_.convergence_tolerance && iter > 5) {
            std::cout << "  Converged at iteration " << iter << ": max divergence = " << max_div << std::endl;
            break;
        }
    }

    sanitize_vector_field(velocity, 0.0);
    apply_boundary_conditions(velocity);
}

void FluidSolver::advect_concentration(const VectorField3D& velocity,
                                        const Grid3D& concentration,
                                        Grid3D& concentration_new,
                                        double dt) const {
    const int nx = static_cast<int>(concentration.nx());
    const int ny = static_cast<int>(concentration.ny());
    const int nz = static_cast<int>(concentration.nz());
    const auto& dims = concentration.dimensions();

    if (concentration_new.nx() != static_cast<size_t>(nx) ||
        concentration_new.ny() != static_cast<size_t>(ny) ||
        concentration_new.nz() != static_cast<size_t>(nz)) {
        concentration_new.allocate(static_cast<size_t>(nx), static_cast<size_t>(ny), static_cast<size_t>(nz));
        concentration_new.dimensions() = dims;
    }
    concentration_new.fill(safe_val(dims.origin_x > 0 ? 0.0 : 0.0, 0.0));

    const double max_disp = static_cast<double>(std::min({nx, ny, nz})) * 0.45;

    #pragma omp parallel for schedule(static, 1)
    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                double u = safe_val(velocity.u(i, j, k), 0.0);
                double v = safe_val(velocity.v(i, j, k), 0.0);
                double w = safe_val(velocity.w(i, j, k), 0.0);

                double dx = u * dt / dims.dx;
                double dy = v * dt / dims.dy;
                double dz = w * dt / dims.dz;

                double disp = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (disp > max_disp) {
                    double scale = max_disp / disp;
                    dx *= scale; dy *= scale; dz *= scale;
                }

                double x = static_cast<double>(i) - dx;
                double y = static_cast<double>(j) - dy;
                double z = static_cast<double>(k) - dz;

                x = std::max(0.5, std::min(static_cast<double>(nx - 1) - 0.5, x));
                y = std::max(0.5, std::min(static_cast<double>(ny - 1) - 0.5, y));
                z = std::max(0.5, std::min(static_cast<double>(nz - 1) - 0.5, z));

                if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
                    concentration_new(i, j, k) = safe_val(concentration(i, j, k), 0.0);
                    continue;
                }

                int ix = static_cast<int>(std::floor(x));
                int iy = static_cast<int>(std::floor(y));
                int iz = static_cast<int>(std::floor(z));

                ix = std::max(0, std::min(nx - 2, ix));
                iy = std::max(0, std::min(ny - 2, iy));
                iz = std::max(0, std::min(nz - 2, iz));

                int ix1 = std::min(ix + 1, nx - 1);
                int iy1 = std::min(iy + 1, ny - 1);
                int iz1 = std::min(iz + 1, nz - 1);

                double tx = std::max(0.0, std::min(1.0, x - ix));
                double ty = std::max(0.0, std::min(1.0, y - iy));
                double tz = std::max(0.0, std::min(1.0, z - iz));

                double c000 = safe_val(concentration(ix, iy, iz), 0.0);
                double c100 = safe_val(concentration(ix1, iy, iz), 0.0);
                double c010 = safe_val(concentration(ix, iy1, iz), 0.0);
                double c110 = safe_val(concentration(ix1, iy1, iz), 0.0);
                double c001 = safe_val(concentration(ix, iy, iz1), 0.0);
                double c101 = safe_val(concentration(ix1, iy, iz1), 0.0);
                double c011 = safe_val(concentration(ix, iy1, iz1), 0.0);
                double c111 = safe_val(concentration(ix1, iy1, iz1), 0.0);

                double c00 = c000 * (1 - tx) + c100 * tx;
                double c10 = c010 * (1 - tx) + c110 * tx;
                double c01 = c001 * (1 - tx) + c101 * tx;
                double c11 = c011 * (1 - tx) + c111 * tx;

                double c0 = c00 * (1 - ty) + c10 * ty;
                double c1 = c01 * (1 - ty) + c11 * ty;

                double c_new = c0 * (1 - tz) + c1 * tz;
                concentration_new(i, j, k) = std::max(0.0, safe_val(c_new, 0.0));
            }
        }
    }

    apply_concentration_boundary_conditions(concentration_new, 0.0);
}

void FluidSolver::diffuse_concentration(const Grid3D& concentration,
                                          Grid3D& concentration_new,
                                          double dt,
                                          double Kx, double Ky, double Kz) const {
    const int nx = static_cast<int>(concentration.nx());
    const int ny = static_cast<int>(concentration.ny());
    const int nz = static_cast<int>(concentration.nz());
    const auto& dims = concentration.dimensions();

    if (concentration_new.nx() != static_cast<size_t>(nx) ||
        concentration_new.ny() != static_cast<size_t>(ny) ||
        concentration_new.nz() != static_cast<size_t>(nz)) {
        concentration_new.allocate(static_cast<size_t>(nx), static_cast<size_t>(ny), static_cast<size_t>(nz));
        concentration_new.dimensions() = dims;
    }

    double dx2 = dims.dx * dims.dx;
    double dy2 = dims.dy * dims.dy;
    double dz2 = dims.dz * dims.dz;

    #pragma omp parallel for schedule(static, 1)
    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                double c = safe_val(concentration(i, j, k), 0.0);
                double c_ip = safe_val(concentration(i + 1, j, k), 0.0);
                double c_im = safe_val(concentration(i - 1, j, k), 0.0);
                double c_jp = safe_val(concentration(i, j + 1, k), 0.0);
                double c_jm = safe_val(concentration(i, j - 1, k), 0.0);
                double c_kp = safe_val(concentration(i, j, k + 1), 0.0);
                double c_km = safe_val(concentration(i, j, k - 1), 0.0);

                double d2c_dx2 = (c_ip - 2.0 * c + c_im) / dx2;
                double d2c_dy2 = (c_jp - 2.0 * c + c_jm) / dy2;
                double d2c_dz2 = (c_kp - 2.0 * c + c_km) / dz2;

                double laplacian = Kx * d2c_dx2 + Ky * d2c_dy2 + Kz * d2c_dz2;
                double c_new = c + dt * laplacian;

                concentration_new(i, j, k) = std::max(0.0, safe_val(c_new, 0.0));
            }
        }
    }

    apply_concentration_boundary_conditions(concentration_new, 0.0);
}

void FluidSolver::apply_source_term(Grid3D& concentration,
                                      const std::vector<AerosolSource>& sources,
                                      double current_time,
                                      double dt) const {
    const int nx = static_cast<int>(concentration.nx());
    const int ny = static_cast<int>(concentration.ny());
    const int nz = static_cast<int>(concentration.nz());
    const auto& dims = concentration.dimensions();

    #pragma omp parallel for schedule(static, 1)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                double total_source = 0.0;

                for (const auto& src : sources) {
                    if (current_time < src.start_time || current_time > src.end_time) {
                        continue;
                    }

                    double dx = (i - src.i) * dims.dx;
                    double dy = (j - src.j) * dims.dy;
                    double dz = (k - src.k) * dims.dz;
                    double r2 = dx * dx + dy * dy + dz * dz;
                    double r = std::sqrt(r2);

                    if (r <= src.radius) {
                        double gaussian = std::exp(-r2 / (2.0 * (src.radius / 3.0) * (src.radius / 3.0)));
                        double strength = src.strength * gaussian;
                        double time_factor = 1.0;
                        if (src.end_time > src.start_time) {
                            double mid_time = 0.5 * (src.start_time + src.end_time);
                            double width = 0.3 * (src.end_time - src.start_time);
                            time_factor = std::exp(-(current_time - mid_time) * (current_time - mid_time) / (2.0 * width * width));
                        }
                        total_source += strength * time_factor;
                    }
                }

                if (total_source > 0.0) {
                    double current_c = safe_val(concentration(i, j, k), 0.0);
                    concentration(i, j, k) = std::max(0.0, current_c + total_source * dt);
                }
            }
        }
    }
}

void FluidSolver::apply_concentration_boundary_conditions(Grid3D& concentration,
                                                             double boundary_value) const {
    const int nx = static_cast<int>(concentration.nx());
    const int ny = static_cast<int>(concentration.ny());
    const int nz = static_cast<int>(concentration.nz());

    if (nx < 2 || ny < 2 || nz < 2) return;

    #pragma omp parallel for schedule(static, 1)
    for (int idx = 0; idx < (2 * ny * nz + 2 * (nx - 2) * nz + 2 * (nx - 2) * (ny - 2)); ++idx) {
        int i, j, k;
        int face = 0;
        int rem = idx;

        if (rem < 2 * ny * nz) {
            face = (rem < ny * nz) ? 0 : 1;
            rem = rem % (ny * nz);
            j = rem / nz;
            k = rem % nz;
            i = (face == 0) ? 0 : (nx - 1);
        } else {
            rem -= 2 * ny * nz;
            if (rem < 2 * (nx - 2) * nz) {
                face = (rem < (nx - 2) * nz) ? 2 : 3;
                rem = rem % ((nx - 2) * nz);
                i = (rem / nz) + 1;
                k = rem % nz;
                j = (face == 2) ? 0 : (ny - 1);
            } else {
                rem -= 2 * (nx - 2) * nz;
                face = (rem < (nx - 2) * (ny - 2)) ? 4 : 5;
                rem = rem % ((nx - 2) * (ny - 2));
                i = (rem / (ny - 2)) + 1;
                j = (rem % (ny - 2)) + 1;
                k = (face == 4) ? 0 : (nz - 1);
            }
        }

        concentration(i, j, k) = std::max(0.0, safe_val(boundary_value, 0.0));
    }
}

void FluidSolver::solve_advection_diffusion(const VectorField3D& velocity,
                                             Grid3D& concentration,
                                             const AdvectionDiffusionConfig& config,
                                             const std::vector<AerosolSource>& sources,
                                             const std::string& output_dir) {
    const int nx = static_cast<int>(concentration.nx());
    const int ny = static_cast<int>(concentration.ny());
    const int nz = static_cast<int>(concentration.nz());

    if (config.num_threads > 0) {
        omp_set_num_threads(config.num_threads);
    }

    concentration.fill(std::max(0.0, safe_val(config.background_concentration, 0.0)));
    sanitize_grid(concentration, 0.0);
    apply_concentration_boundary_conditions(concentration, 0.0);

    Grid3D conc_prev(static_cast<size_t>(nx), static_cast<size_t>(ny), static_cast<size_t>(nz));
    conc_prev.dimensions() = concentration.dimensions();
    Grid3D conc_temp(static_cast<size_t>(nx), static_cast<size_t>(ny), static_cast<size_t>(nz));
    conc_temp.dimensions() = concentration.dimensions();

    double current_time = 0.0;
    double next_output_time = 0.0;
    int frame_count = 0;

    std::cout << "\n========================================" << std::endl;
    std::cout << "  Advection-Diffusion Solver" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Grid: " << nx << " x " << ny << " x " << nz << std::endl;
    std::cout << "  Time step: " << config.time_step << " s" << std::endl;
    std::cout << "  Total time: " << config.total_time << " s (" 
              << config.total_time / 3600.0 << " h)" << std::endl;
    std::cout << "  Output interval: " << config.output_interval << " s (" 
              << config.output_interval / 3600.0 << " h)" << std::endl;
    std::cout << "  Diffusion: Kx=" << config.diffusion_coeff_x 
              << ", Ky=" << config.diffusion_coeff_y 
              << ", Kz=" << config.diffusion_coeff_z << " m²/s" << std::endl;
    std::cout << "  Sources: " << sources.size() << std::endl;
    std::cout << "========================================" << std::endl;

    while (current_time < config.total_time) {
        double dt = std::min(config.time_step, config.total_time - current_time);

        conc_prev.data_vector().swap(concentration.data_vector());
        sanitize_grid(conc_prev, 0.0);

        advect_concentration(velocity, conc_prev, conc_temp, dt);
        apply_source_term(conc_temp, sources, current_time, dt);
        diffuse_concentration(conc_temp, concentration, dt, 
                              config.diffusion_coeff_x,
                              config.diffusion_coeff_y,
                              config.diffusion_coeff_z);

        sanitize_grid(concentration, 0.0);
        apply_concentration_boundary_conditions(concentration, 0.0);

        current_time += dt;

        if (current_time >= next_output_time - 1e-10) {
            double max_c = 0.0;
            double total_mass = 0.0;
            int nans = count_nan(concentration);

            #pragma omp parallel for reduction(max:max_c) reduction(+:total_mass) schedule(static)
            for (int k = 0; k < nz; ++k) {
                for (int j = 0; j < ny; ++j) {
                    for (int i = 0; i < nx; ++i) {
                        double c = safe_val(concentration(i, j, k), 0.0);
                        if (c > max_c) max_c = c;
                        total_mass += c;
                    }
                }
            }

            double cell_volume = concentration.dimensions().dx * 
                                 concentration.dimensions().dy * 
                                 concentration.dimensions().dz;
            total_mass *= cell_volume;

            int hours = static_cast<int>(current_time / 3600.0);
            int minutes = static_cast<int>((current_time - hours * 3600.0) / 60.0);
            int seconds = static_cast<int>(current_time - hours * 3600.0 - minutes * 60.0);

            printf("  T=%02d:%02d:%02d (%.1fs) | max_C=%.2e | total_mass=%.2e | NaN=%d\n",
                   hours, minutes, seconds, current_time, max_c, total_mass, nans);

            if (nans > 0) {
                std::cerr << "  [WARNING] NaN detected at t=" << current_time 
                          << ", sanitizing..." << std::endl;
                sanitize_grid(concentration, 0.0);
            }

            next_output_time += config.output_interval;
            frame_count++;
        }
    }

    std::cout << "========================================" << std::endl;
    std::cout << "  Simulation completed. Frames: " << frame_count << std::endl;
    std::cout << "========================================" << std::endl;
}

}  // namespace mfd
