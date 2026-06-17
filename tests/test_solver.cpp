#include "fluid_solver.h"
#include "grid3d.h"
#include <omp.h>
#include <iostream>
#include <cassert>
#include <cmath>
#include <chrono>

using namespace mfd;

void test_divergence() {
    std::cout << "\n--- Divergence Test ---" << std::endl;

    const size_t N = 64;
    VectorField3D velocity(N, N, N, 1.0, 1.0, 1.0);

    for (size_t k = 0; k < N; ++k) {
        for (size_t j = 0; j < N; ++j) {
            for (size_t i = 0; i < N; ++i) {
                velocity.u(i, j, k) = static_cast<double>(i);
                velocity.v(i, j, k) = static_cast<double>(j);
                velocity.w(i, j, k) = static_cast<double>(k);
            }
        }
    }

    Grid3D divergence;
    FluidSolver solver;
    solver.compute_divergence(velocity, divergence);

    double expected_div = 3.0;
    int interior_ok = 0;
    int interior_total = 0;
    for (size_t k = 1; k < N - 1; ++k) {
        for (size_t j = 1; j < N - 1; ++j) {
            for (size_t i = 1; i < N - 1; ++i) {
                interior_total++;
                if (std::abs(divergence(i, j, k) - expected_div) < 0.01) {
                    interior_ok++;
                }
            }
        }
    }
    std::cout << "Divergence accuracy: " << interior_ok << "/" << interior_total
              << " interior points within tolerance" << std::endl;
    assert(interior_ok > interior_total * 0.9);
    std::cout << "[PASS] Divergence computation" << std::endl;
}

void test_vorticity() {
    std::cout << "\n--- Vorticity Test ---" << std::endl;

    const size_t N = 64;
    VectorField3D velocity(N, N, N, 1.0, 1.0, 1.0);

    for (size_t k = 0; k < N; ++k) {
        for (size_t j = 0; j < N; ++j) {
            for (size_t i = 0; i < N; ++i) {
                double x = static_cast<double>(i) - N / 2.0;
                double y = static_cast<double>(j) - N / 2.0;
                velocity.u(i, j, k) = -y;
                velocity.v(i, j, k) = x;
                velocity.w(i, j, k) = 0.0;
            }
        }
    }

    VectorField3D vorticity;
    Grid3D vort_mag;
    FluidSolver solver;
    solver.compute_vorticity(velocity, vorticity);
    solver.compute_vorticity_magnitude(vorticity, vort_mag);

    int interior_ok = 0;
    int interior_total = 0;
    for (size_t k = 1; k < N - 1; ++k) {
        for (size_t j = 1; j < N - 1; ++j) {
            for (size_t i = 1; i < N - 1; ++i) {
                interior_total++;
                double expected_wz = 2.0;
                if (std::abs(vorticity.w(i, j, k) - expected_wz) < 0.01 &&
                    std::abs(vorticity.u(i, j, k)) < 0.01 &&
                    std::abs(vorticity.v(i, j, k)) < 0.01) {
                    interior_ok++;
                }
            }
        }
    }
    std::cout << "Vorticity accuracy: " << interior_ok << "/" << interior_total
              << " interior points within tolerance" << std::endl;
    assert(interior_ok > interior_total * 0.9);
    std::cout << "[PASS] Vorticity computation" << std::endl;
}

void test_navier_stokes() {
    std::cout << "\n--- Navier-Stokes Solver Test ---" << std::endl;

    const size_t N = 50;
    VectorField3D velocity(N, N, N, 100.0, 100.0, 100.0);
    Grid3D reflectivity(N, N, N, 100.0, 100.0, 100.0);

    for (size_t k = 0; k < N; ++k) {
        for (size_t j = 0; j < N; ++j) {
            for (size_t i = 0; i < N; ++i) {
                double x = static_cast<double>(i) - N / 2.0;
                double y = static_cast<double>(j) - N / 2.0;
                double z = static_cast<double>(k) - N / 2.0;
                double r2 = x * x + y * y + z * z;
                velocity.u(i, j, k) = 10.0 * std::exp(-r2 / (N * N * 4.0));
                velocity.v(i, j, k) = 5.0 * std::exp(-r2 / (N * N * 4.0));
                velocity.w(i, j, k) = 2.0 * std::exp(-r2 / (N * N * 4.0));
                reflectivity(i, j, k) = 30.0 * std::exp(-r2 / (N * N * 2.0));
            }
        }
    }

    SolverConfig config;
    config.num_threads = 4;
    config.time_step = 0.01;
    config.max_iterations = 30;
    FluidSolver solver(config);

    auto t0 = std::chrono::high_resolution_clock::now();
    solver.solve_navier_stokes_simplified(velocity, reflectivity, 30);
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Navier-Stokes solver time: " << elapsed_ms << " ms" << std::endl;

    Grid3D divergence;
    solver.compute_divergence(velocity, divergence);
    double max_div = 0.0;
    for (size_t k = 0; k < divergence.nz(); ++k) {
        for (size_t j = 0; j < divergence.ny(); ++j) {
            for (size_t i = 0; i < divergence.nx(); ++i) {
                max_div = std::max(max_div, std::abs(divergence(i, j, k)));
            }
        }
    }
    std::cout << "Final max divergence: " << max_div << std::endl;
    assert(max_div < 0.1);
    std::cout << "[PASS] Navier-Stokes simplified solver" << std::endl;
}

void test_incompressibility() {
    std::cout << "\n--- Incompressibility Enforcement Test ---" << std::endl;

    const size_t N = 32;
    VectorField3D velocity(N, N, N, 1.0, 1.0, 1.0);
    velocity.u.fill(1.0);
    velocity.v.fill(1.0);
    velocity.w.fill(1.0);

    FluidSolver solver;
    Grid3D divergence_before, divergence_after;
    solver.compute_divergence(velocity, divergence_before);

    double max_div_before = 0.0;
    for (size_t i = 0; i < divergence_before.size(); ++i) {
        max_div_before = std::max(max_div_before, std::abs(divergence_before.data()[i]));
    }

    solver.enforce_incompressibility(velocity, 50);
    solver.compute_divergence(velocity, divergence_after);

    double max_div_after = 0.0;
    for (size_t i = 0; i < divergence_after.size(); ++i) {
        max_div_after = std::max(max_div_after, std::abs(divergence_after.data()[i]));
    }

    std::cout << "Max divergence before: " << max_div_before << std::endl;
    std::cout << "Max divergence after: " << max_div_after << std::endl;
    assert(max_div_after < max_div_before * 0.1);
    std::cout << "[PASS] Incompressibility enforcement" << std::endl;
}

int main() {
    std::cout << "=== Fluid Solver Tests ===" << std::endl;
    std::cout << "OpenMP max threads: " << omp_get_max_threads() << std::endl;

    test_divergence();
    test_vorticity();
    test_incompressibility();
    test_navier_stokes();

    std::cout << "\n=== All Fluid Solver Tests PASSED ===" << std::endl;
    return 0;
}
