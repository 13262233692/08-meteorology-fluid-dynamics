#ifndef MFD_FLUID_SOLVER_H
#define MFD_FLUID_SOLVER_H

#include "grid3d.h"

namespace mfd {

struct SolverConfig {
    double kinematic_viscosity;
    double density;
    double time_step;
    int max_iterations;
    double convergence_tolerance;
    int num_threads;

    SolverConfig()
        : kinematic_viscosity(1.5e-5)
        , density(1.225)
        , time_step(0.01)
        , max_iterations(100)
        , convergence_tolerance(1e-6)
        , num_threads(0) {}
};

class FluidSolver {
public:
    explicit FluidSolver(const SolverConfig& config = SolverConfig());
    ~FluidSolver();

    void compute_divergence(const VectorField3D& velocity, Grid3D& divergence) const;

    void compute_vorticity(const VectorField3D& velocity, VectorField3D& vorticity) const;

    void compute_vorticity_magnitude(const VectorField3D& vorticity, Grid3D& magnitude) const;

    void solve_navier_stokes_simplified(VectorField3D& velocity,
                                        const Grid3D& reflectivity,
                                        int iterations = -1);

    void compute_velocity_from_radial(const Grid3D& radial_wind,
                                      const Grid3D& reflectivity,
                                      VectorField3D& velocity,
                                      double elevation = 0.0,
                                      double azimuth = 0.0);

    void apply_boundary_conditions(VectorField3D& velocity) const;

    void enforce_incompressibility(VectorField3D& velocity, int iterations = 50);

    void sanitize_grid(Grid3D& grid, double fill_value = 0.0) const;
    void sanitize_vector_field(VectorField3D& vf, double fill_value = 0.0) const;
    size_t count_nan(const Grid3D& grid) const;

    void set_config(const SolverConfig& config) { config_ = config; }
    const SolverConfig& config() const { return config_; }

private:
    SolverConfig config_;

    void compute_pressure_poisson(const VectorField3D& velocity, Grid3D& pressure) const;
    void apply_pressure_gradient(VectorField3D& velocity, const Grid3D& pressure) const;
    void diffuse_velocity(VectorField3D& velocity, double dt) const;
    void advect_velocity(VectorField3D& velocity, double dt) const;
    void add_buoyancy_term(VectorField3D& velocity, const Grid3D& reflectivity) const;

    inline double ddx_central(const Grid3D& f, size_t i, size_t j, size_t k) const;
    inline double ddy_central(const Grid3D& f, size_t i, size_t j, size_t k) const;
    inline double ddz_central(const Grid3D& f, size_t i, size_t j, size_t k) const;
    inline double laplacian(const Grid3D& f, size_t i, size_t j, size_t k) const;

    inline bool is_valid(double val) const;
    inline double safe_val(double val, double fallback = 0.0) const;
};

}  // namespace mfd

#endif  // MFD_FLUID_SOLVER_H
