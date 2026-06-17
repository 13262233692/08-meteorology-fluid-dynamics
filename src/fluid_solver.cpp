#include "fluid_solver.h"
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace mfd {

FluidSolver::FluidSolver(const SolverConfig& config)
    : config_(config) {
    if (config_.num_threads > 0) {
        omp_set_num_threads(config_.num_threads);
    }
}

FluidSolver::~FluidSolver() = default;

inline double FluidSolver::ddx_central(const Grid3D& f, size_t i, size_t j, size_t k) const {
    size_t ip = std::min(i + 1, f.nx() - 1);
    size_t im = (i > 0) ? (i - 1) : 0;
    return (f(ip, j, k) - f(im, j, k)) / (2.0 * f.dimensions().dx);
}

inline double FluidSolver::ddy_central(const Grid3D& f, size_t i, size_t j, size_t k) const {
    size_t jp = std::min(j + 1, f.ny() - 1);
    size_t jm = (j > 0) ? (j - 1) : 0;
    return (f(i, jp, k) - f(i, jm, k)) / (2.0 * f.dimensions().dy);
}

inline double FluidSolver::ddz_central(const Grid3D& f, size_t i, size_t j, size_t k) const {
    size_t kp = std::min(k + 1, f.nz() - 1);
    size_t km = (k > 0) ? (k - 1) : 0;
    return (f(i, j, kp) - f(i, j, km)) / (2.0 * f.dimensions().dz);
}

inline double FluidSolver::laplacian(const Grid3D& f, size_t i, size_t j, size_t k) const {
    size_t ip = std::min(i + 1, f.nx() - 1);
    size_t im = (i > 0) ? (i - 1) : 0;
    size_t jp = std::min(j + 1, f.ny() - 1);
    size_t jm = (j > 0) ? (j - 1) : 0;
    size_t kp = std::min(k + 1, f.nz() - 1);
    size_t km = (k > 0) ? (k - 1) : 0;

    double dx2 = f.dimensions().dx * f.dimensions().dx;
    double dy2 = f.dimensions().dy * f.dimensions().dy;
    double dz2 = f.dimensions().dz * f.dimensions().dz;

    return (f(ip, j, k) - 2.0 * f(i, j, k) + f(im, j, k)) / dx2 +
           (f(i, jp, k) - 2.0 * f(i, j, k) + f(i, jm, k)) / dy2 +
           (f(i, j, kp) - 2.0 * f(i, j, k) + f(i, j, km)) / dz2;
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

    #pragma omp parallel for schedule(dynamic, 8)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                double dudx = ddx_central(velocity.u, i, j, k);
                double dvdy = ddy_central(velocity.v, i, j, k);
                double dwdz = ddz_central(velocity.w, i, j, k);
                divergence(i, j, k) = dudx + dvdy + dwdz;
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

    #pragma omp parallel for schedule(dynamic, 8)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                double dwdy = ddy_central(velocity.w, i, j, k);
                double dvdz = ddz_central(velocity.v, i, j, k);
                vorticity.u(i, j, k) = dwdy - dvdz;

                double dudz = ddz_central(velocity.u, i, j, k);
                double dwdx = ddx_central(velocity.w, i, j, k);
                vorticity.v(i, j, k) = dudz - dwdx;

                double dvdx = ddx_central(velocity.v, i, j, k);
                double dudy = ddy_central(velocity.u, i, j, k);
                vorticity.w(i, j, k) = dvdx - dudy;
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

    #pragma omp parallel for schedule(dynamic, 8)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                double wx = vorticity.u(i, j, k);
                double wy = vorticity.v(i, j, k);
                double wz = vorticity.w(i, j, k);
                magnitude(i, j, k) = std::sqrt(wx * wx + wy * wy + wz * wz);
            }
        }
    }
}

void FluidSolver::apply_boundary_conditions(VectorField3D& velocity) const {
    const int nx = static_cast<int>(velocity.nx());
    const int ny = static_cast<int>(velocity.ny());
    const int nz = static_cast<int>(velocity.nz());

    if (nx < 2 || ny < 2 || nz < 2) return;

    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                velocity.u(0, j, k) = 0.0;
                velocity.v(0, j, k) = 0.0;
                velocity.w(0, j, k) = 0.0;
                velocity.u(nx - 1, j, k) = 0.0;
                velocity.v(nx - 1, j, k) = 0.0;
                velocity.w(nx - 1, j, k) = 0.0;
            }
        }

        #pragma omp for schedule(static)
        for (int i = 0; i < nx; ++i) {
            for (int k = 0; k < nz; ++k) {
                velocity.u(i, 0, k) = 0.0;
                velocity.v(i, 0, k) = 0.0;
                velocity.w(i, 0, k) = 0.0;
                velocity.u(i, ny - 1, k) = 0.0;
                velocity.v(i, ny - 1, k) = 0.0;
                velocity.w(i, ny - 1, k) = 0.0;
            }
        }

        #pragma omp for schedule(static)
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                velocity.u(i, j, 0) = 0.0;
                velocity.v(i, j, 0) = 0.0;
                velocity.w(i, j, 0) = 0.0;
                velocity.u(i, j, nz - 1) = 0.0;
                velocity.v(i, j, nz - 1) = 0.0;
                velocity.w(i, j, nz - 1) = 0.0;
            }
        }
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

    #pragma omp parallel for schedule(dynamic, 8)
    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                temp.u(i, j, k) = velocity.u(i, j, k) + nu * dt * laplacian(velocity.u, i, j, k);
                temp.v(i, j, k) = velocity.v(i, j, k) + nu * dt * laplacian(velocity.v, i, j, k);
                temp.w(i, j, k) = velocity.w(i, j, k) + nu * dt * laplacian(velocity.w, i, j, k);
            }
        }
    }

    velocity.u.data_vector().swap(temp.u.data_vector());
    velocity.v.data_vector().swap(temp.v.data_vector());
    velocity.w.data_vector().swap(temp.w.data_vector());
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

    #pragma omp parallel for schedule(dynamic, 4)
    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                double u = velocity.u(i, j, k);
                double v = velocity.v(i, j, k);
                double w = velocity.w(i, j, k);

                double x = static_cast<double>(i) - u * dt / dims.dx;
                double y = static_cast<double>(j) - v * dt / dims.dy;
                double z = static_cast<double>(k) - w * dt / dims.dz;

                x = std::max(0.0, std::min(static_cast<double>(nx - 1), x));
                y = std::max(0.0, std::min(static_cast<double>(ny - 1), y));
                z = std::max(0.0, std::min(static_cast<double>(nz - 1), z));

                int ix = static_cast<int>(std::floor(x));
                int iy = static_cast<int>(std::floor(y));
                int iz = static_cast<int>(std::floor(z));
                int ix1 = std::min(ix + 1, nx - 1);
                int iy1 = std::min(iy + 1, ny - 1);
                int iz1 = std::min(iz + 1, nz - 1);

                double tx = x - ix;
                double ty = y - iy;
                double tz = z - iz;

                auto interp = [&](const Grid3D& f) -> double {
                    double c00 = f(ix, iy, iz) * (1 - tx) + f(ix1, iy, iz) * tx;
                    double c10 = f(ix, iy1, iz) * (1 - tx) + f(ix1, iy1, iz) * tx;
                    double c01 = f(ix, iy, iz1) * (1 - tx) + f(ix1, iy, iz1) * tx;
                    double c11 = f(ix, iy1, iz1) * (1 - tx) + f(ix1, iy1, iz1) * tx;
                    double c0 = c00 * (1 - ty) + c10 * ty;
                    double c1 = c01 * (1 - ty) + c11 * ty;
                    return c0 * (1 - tz) + c1 * tz;
                };

                temp.u(i, j, k) = interp(velocity.u);
                temp.v(i, j, k) = interp(velocity.v);
                temp.w(i, j, k) = interp(velocity.w);
            }
        }
    }

    velocity.u.data_vector().swap(temp.u.data_vector());
    velocity.v.data_vector().swap(temp.v.data_vector());
    velocity.w.data_vector().swap(temp.w.data_vector());
}

void FluidSolver::add_buoyancy_term(VectorField3D& velocity, const Grid3D& reflectivity) const {
    const int nx = static_cast<int>(velocity.nx());
    const int ny = static_cast<int>(velocity.ny());
    const int nz = static_cast<int>(velocity.nz());
    const double g = 9.81;
    const double rho0 = config_.density;

    #pragma omp parallel for schedule(dynamic, 8)
    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                double dbz = reflectivity(i, j, k);
                if (dbz > 10.0) {
                    double buoyancy = g * (dbz - 10.0) / (50.0 * rho0);
                    velocity.w(i, j, k) += buoyancy * config_.time_step;
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

    #pragma omp parallel for schedule(dynamic, 8)
    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                double dudx = ddx_central(velocity.u, i, j, k);
                double dvdy = ddy_central(velocity.v, i, j, k);
                double dwdz = ddz_central(velocity.w, i, j, k);
                rhs(i, j, k) = (dudx + dvdy + dwdz) / config_.time_step;
            }
        }
    }

    pressure.fill(0.0);
    Grid3D pressure_new(static_cast<size_t>(nx), static_cast<size_t>(ny), static_cast<size_t>(nz),
                         dims.dx, dims.dy, dims.dz);

    double dx2 = dims.dx * dims.dx;
    double dy2 = dims.dy * dims.dy;
    double dz2 = dims.dz * dims.dz;
    double inv_coeff = 1.0 / (2.0 / dx2 + 2.0 / dy2 + 2.0 / dz2);

    const int jacobi_iterations = 80;
    for (int iter = 0; iter < jacobi_iterations; ++iter) {
        #pragma omp parallel for schedule(dynamic, 8)
        for (int k = 1; k < nz - 1; ++k) {
            for (int j = 1; j < ny - 1; ++j) {
                for (int i = 1; i < nx - 1; ++i) {
                    double sum = (pressure(i + 1, j, k) + pressure(i - 1, j, k)) / dx2 +
                                 (pressure(i, j + 1, k) + pressure(i, j - 1, k)) / dy2 +
                                 (pressure(i, j, k + 1) + pressure(i, j, k - 1)) / dz2;
                    pressure_new(i, j, k) = (sum - rhs(i, j, k)) * inv_coeff;
                }
            }
        }

        #pragma omp parallel for schedule(dynamic, 16)
        for (int k = 1; k < nz - 1; ++k) {
            for (int j = 1; j < ny - 1; ++j) {
                for (int i = 1; i < nx - 1; ++i) {
                    pressure(i, j, k) = pressure_new(i, j, k);
                }
            }
        }
    }
}

void FluidSolver::apply_pressure_gradient(VectorField3D& velocity, const Grid3D& pressure) const {
    const int nx = static_cast<int>(velocity.nx());
    const int ny = static_cast<int>(velocity.ny());
    const int nz = static_cast<int>(velocity.nz());
    const double rho = config_.density;

    #pragma omp parallel for schedule(dynamic, 8)
    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                velocity.u(i, j, k) -= config_.time_step / rho * ddx_central(pressure, i, j, k);
                velocity.v(i, j, k) -= config_.time_step / rho * ddy_central(pressure, i, j, k);
                velocity.w(i, j, k) -= config_.time_step / rho * ddz_central(pressure, i, j, k);
            }
        }
    }
}

void FluidSolver::enforce_incompressibility(VectorField3D& velocity, int iterations) {
    Grid3D pressure;
    const auto& dims = velocity.u.dimensions();
    pressure.allocate(dims.nx, dims.ny, dims.nz);
    pressure.dimensions() = dims;

    for (int iter = 0; iter < iterations; ++iter) {
        compute_pressure_poisson(velocity, pressure);
        apply_pressure_gradient(velocity, pressure);
    }
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

    #pragma omp parallel for schedule(dynamic, 8)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                double vr = radial_wind(i, j, k);
                if (std::isnan(vr) || std::isinf(vr)) {
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

    enforce_incompressibility(velocity, 30);
    (void)reflectivity;
}

void FluidSolver::solve_navier_stokes_simplified(VectorField3D& velocity,
                                                   const Grid3D& reflectivity,
                                                   int iterations) {
    int iters = (iterations > 0) ? iterations : config_.max_iterations;

    apply_boundary_conditions(velocity);

    for (int iter = 0; iter < iters; ++iter) {
        add_buoyancy_term(velocity, reflectivity);

        advect_velocity(velocity, config_.time_step);

        diffuse_velocity(velocity, config_.time_step);

        enforce_incompressibility(velocity, 20);

        apply_boundary_conditions(velocity);

        Grid3D div;
        compute_divergence(velocity, div);

        const int nx = static_cast<int>(div.nx());
        const int ny = static_cast<int>(div.ny());
        const int nz = static_cast<int>(div.nz());

        double max_div = 0.0;
        #pragma omp parallel for reduction(max:max_div) schedule(dynamic)
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    double d = std::abs(div(i, j, k));
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
}

}  // namespace mfd
