#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <omp.h>
#include "grid3d.h"
#include "netcdf_reader.h"
#include "fluid_solver.h"

namespace py = pybind11;
using namespace mfd;

#ifdef _MSC_VER
using ssize_t = py::ssize_t;
#endif

namespace {

py::array_t<double> grid3d_to_numpy(Grid3D& grid) {
    auto shape = std::vector<ssize_t>{
        static_cast<ssize_t>(grid.nz()),
        static_cast<ssize_t>(grid.ny()),
        static_cast<ssize_t>(grid.nx())
    };
    auto strides = std::vector<ssize_t>{
        static_cast<ssize_t>(grid.ny() * grid.nx() * sizeof(double)),
        static_cast<ssize_t>(grid.nx() * sizeof(double)),
        static_cast<ssize_t>(sizeof(double))
    };
    return py::array_t<double>(
        shape, strides, grid.data(), py::cast(grid)
    );
}

py::array_t<double> grid3d_to_numpy_const(const Grid3D& grid) {
    auto shape = std::vector<ssize_t>{
        static_cast<ssize_t>(grid.nz()),
        static_cast<ssize_t>(grid.ny()),
        static_cast<ssize_t>(grid.nx())
    };
    auto strides = std::vector<ssize_t>{
        static_cast<ssize_t>(grid.ny() * grid.nx() * sizeof(double)),
        static_cast<ssize_t>(grid.nx() * sizeof(double)),
        static_cast<ssize_t>(sizeof(double))
    };
    return py::array_t<double>(
        shape, strides, const_cast<double*>(grid.data()),
        py::cast(static_cast<const Grid3D*>(&grid))
    );
}

void numpy_to_grid3d(Grid3D& grid, py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 3) {
        throw std::runtime_error("Expected 3D numpy array");
    }
    size_t nz = static_cast<size_t>(buf.shape[0]);
    size_t ny = static_cast<size_t>(buf.shape[1]);
    size_t nx = static_cast<size_t>(buf.shape[2]);
    if (grid.nx() != nx || grid.ny() != ny || grid.nz() != nz) {
        grid.allocate(nx, ny, nz);
    }
    std::memcpy(grid.data(), buf.ptr, nx * ny * nz * sizeof(double));
}

}  // namespace

PYBIND11_MODULE(mfd, m) {
    m.doc() = "Meteorology Fluid Dynamics High-Performance Computing Engine";

    py::class_<GridDimensions>(m, "GridDimensions")
        .def(py::init<>())
        .def_readwrite("nx", &GridDimensions::nx)
        .def_readwrite("ny", &GridDimensions::ny)
        .def_readwrite("nz", &GridDimensions::nz)
        .def_readwrite("dx", &GridDimensions::dx)
        .def_readwrite("dy", &GridDimensions::dy)
        .def_readwrite("dz", &GridDimensions::dz)
        .def_readwrite("origin_x", &GridDimensions::origin_x)
        .def_readwrite("origin_y", &GridDimensions::origin_y)
        .def_readwrite("origin_z", &GridDimensions::origin_z);

    py::class_<Grid3D, std::shared_ptr<Grid3D>>(m, "Grid3D", py::buffer_protocol())
        .def(py::init<>())
        .def(py::init<size_t, size_t, size_t, double, double, double, double, double, double>(),
             py::arg("nx"), py::arg("ny"), py::arg("nz"),
             py::arg("dx") = 1.0, py::arg("dy") = 1.0, py::arg("dz") = 1.0,
             py::arg("ox") = 0.0, py::arg("oy") = 0.0, py::arg("oz") = 0.0)
        .def_property_readonly("nx", &Grid3D::nx)
        .def_property_readonly("ny", &Grid3D::ny)
        .def_property_readonly("nz", &Grid3D::nz)
        .def_property_readonly("size", &Grid3D::size)
        .def_property("dimensions",
                     py::overload_cast<>(&Grid3D::dimensions, py::const_),
                     [](Grid3D& g, const GridDimensions& d) { g.dimensions() = d; })
        .def("allocate", &Grid3D::allocate)
        .def("clear", &Grid3D::clear)
        .def("fill", &Grid3D::fill)
        .def("copy_from", [](Grid3D& g, py::array_t<double> arr) {
            py::buffer_info buf = arr.request();
            if (buf.ndim != 1) {
                throw std::runtime_error("Expected 1D array for copy_from");
            }
            g.copy_from(static_cast<double*>(buf.ptr),
                        static_cast<size_t>(buf.shape[0]));
        })
        .def("__call__", [](Grid3D& g, size_t i, size_t j, size_t k) -> double& {
            return g(i, j, k);
        }, py::return_value_policy::reference_internal)
        .def("__getitem__", [](const Grid3D& g, std::tuple<size_t, size_t, size_t> idx) {
            return g(std::get<0>(idx), std::get<1>(idx), std::get<2>(idx));
        })
        .def("__setitem__", [](Grid3D& g, std::tuple<size_t, size_t, size_t> idx, double val) {
            g(std::get<0>(idx), std::get<1>(idx), std::get<2>(idx)) = val;
        })
        .def("to_numpy", [](Grid3D& g) { return grid3d_to_numpy(g); },
             "Get numpy array view (zero-copy, shared memory)")
        .def("from_numpy", [](Grid3D& g, py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            numpy_to_grid3d(g, arr);
        }, "Copy data from numpy array")
        .def_property_readonly("data_ptr", [](const Grid3D& g) {
            return reinterpret_cast<uint64_t>(g.data());
        }, "Raw pointer to underlying data (as integer)")
        .def("swap", [](Grid3D& g, Grid3D& other) {
            g.data_vector().swap(other.data_vector());
        }, py::arg("other"), "Efficiently swap data with another Grid3D")
        .def_buffer([](Grid3D& g) -> py::buffer_info {
            return py::buffer_info(
                g.data(), sizeof(double), py::format_descriptor<double>::format(),
                3,
                {static_cast<ssize_t>(g.nz()), static_cast<ssize_t>(g.ny()), static_cast<ssize_t>(g.nx())},
                {static_cast<ssize_t>(g.ny() * g.nx() * sizeof(double)),
                 static_cast<ssize_t>(g.nx() * sizeof(double)),
                 static_cast<ssize_t>(sizeof(double))}
            );
        });

    py::class_<VectorField3D, std::shared_ptr<VectorField3D>>(m, "VectorField3D")
        .def(py::init<>())
        .def(py::init<size_t, size_t, size_t, double, double, double>(),
             py::arg("nx"), py::arg("ny"), py::arg("nz"),
             py::arg("dx") = 1.0, py::arg("dy") = 1.0, py::arg("dz") = 1.0)
        .def_readwrite("u", &VectorField3D::u)
        .def_readwrite("v", &VectorField3D::v)
        .def_readwrite("w", &VectorField3D::w)
        .def_property_readonly("nx", &VectorField3D::nx)
        .def_property_readonly("ny", &VectorField3D::ny)
        .def_property_readonly("nz", &VectorField3D::nz)
        .def("component", py::overload_cast<int>(&VectorField3D::component),
             py::return_value_policy::reference_internal)
        .def("allocate", &VectorField3D::allocate)
        .def("to_numpy_tuple", [](VectorField3D& vf) {
            return py::make_tuple(grid3d_to_numpy(vf.u),
                                  grid3d_to_numpy(vf.v),
                                  grid3d_to_numpy(vf.w));
        }, "Get (u, v, w) as tuple of numpy arrays");

    py::class_<RadarVolumeScan>(m, "RadarVolumeScan")
        .def(py::init<>())
        .def_readwrite("reflectivity", &RadarVolumeScan::reflectivity)
        .def_readwrite("radial_velocity", &RadarVolumeScan::radial_velocity)
        .def_readwrite("dims", &RadarVolumeScan::dims)
        .def_readwrite("elevation_angles", &RadarVolumeScan::elevation_angles)
        .def_readwrite("azimuth_angles", &RadarVolumeScan::azimuth_angles)
        .def_readwrite("range_gates", &RadarVolumeScan::range_gates)
        .def_readwrite("timestamp", &RadarVolumeScan::timestamp)
        .def_readwrite("radar_name", &RadarVolumeScan::radar_name);

    py::class_<NetCDFReader>(m, "NetCDFReader")
        .def(py::init<>())
        .def("read_radar_volume", &NetCDFReader::read_radar_volume)
        .def("read_3d_variable", &NetCDFReader::read_3d_variable)
        .def("read_grid_dimensions", &NetCDFReader::read_grid_dimensions)
        .def_property("reflectivity_varname",
                     &NetCDFReader::reflectivity_varname,
                     &NetCDFReader::set_reflectivity_varname)
        .def_property("velocity_varname",
                     &NetCDFReader::velocity_varname,
                     &NetCDFReader::set_velocity_varname)
        .def_property("radial_varname",
                     &NetCDFReader::radial_varname,
                     &NetCDFReader::set_radial_varname);

    py::class_<SolverConfig>(m, "SolverConfig")
        .def(py::init<>())
        .def_readwrite("kinematic_viscosity", &SolverConfig::kinematic_viscosity)
        .def_readwrite("density", &SolverConfig::density)
        .def_readwrite("time_step", &SolverConfig::time_step)
        .def_readwrite("max_iterations", &SolverConfig::max_iterations)
        .def_readwrite("convergence_tolerance", &SolverConfig::convergence_tolerance)
        .def_readwrite("num_threads", &SolverConfig::num_threads);

    py::class_<AdvectionDiffusionConfig>(m, "AdvectionDiffusionConfig")
        .def(py::init<>())
        .def_readwrite("diffusion_coeff_x", &AdvectionDiffusionConfig::diffusion_coeff_x)
        .def_readwrite("diffusion_coeff_y", &AdvectionDiffusionConfig::diffusion_coeff_y)
        .def_readwrite("diffusion_coeff_z", &AdvectionDiffusionConfig::diffusion_coeff_z)
        .def_readwrite("time_step", &AdvectionDiffusionConfig::time_step)
        .def_readwrite("total_time", &AdvectionDiffusionConfig::total_time)
        .def_readwrite("output_interval", &AdvectionDiffusionConfig::output_interval)
        .def_readwrite("source_strength", &AdvectionDiffusionConfig::source_strength)
        .def_readwrite("source_decay_rate", &AdvectionDiffusionConfig::source_decay_rate)
        .def_readwrite("background_concentration", &AdvectionDiffusionConfig::background_concentration)
        .def_readwrite("source_grid_i", &AdvectionDiffusionConfig::source_grid_i)
        .def_readwrite("source_grid_j", &AdvectionDiffusionConfig::source_grid_j)
        .def_readwrite("source_grid_k", &AdvectionDiffusionConfig::source_grid_k)
        .def_readwrite("source_radius", &AdvectionDiffusionConfig::source_radius)
        .def_readwrite("num_threads", &AdvectionDiffusionConfig::num_threads);

    py::class_<AerosolSource>(m, "AerosolSource")
        .def(py::init<>())
        .def_readwrite("i", &AerosolSource::i)
        .def_readwrite("j", &AerosolSource::j)
        .def_readwrite("k", &AerosolSource::k)
        .def_readwrite("strength", &AerosolSource::strength)
        .def_readwrite("start_time", &AerosolSource::start_time)
        .def_readwrite("end_time", &AerosolSource::end_time)
        .def_readwrite("radius", &AerosolSource::radius)
        .def_readwrite("name", &AerosolSource::name);

    py::class_<FluidSolver>(m, "FluidSolver")
        .def(py::init<const SolverConfig&>(), py::arg("config") = SolverConfig())
        .def("compute_divergence", &FluidSolver::compute_divergence,
             py::arg("velocity"), py::arg("divergence"))
        .def("compute_vorticity", &FluidSolver::compute_vorticity,
             py::arg("velocity"), py::arg("vorticity"))
        .def("compute_vorticity_magnitude", &FluidSolver::compute_vorticity_magnitude,
             py::arg("vorticity"), py::arg("magnitude"))
        .def("solve_navier_stokes_simplified", &FluidSolver::solve_navier_stokes_simplified,
             py::arg("velocity"), py::arg("reflectivity"), py::arg("iterations") = -1)
        .def("compute_velocity_from_radial", &FluidSolver::compute_velocity_from_radial,
             py::arg("radial_wind"), py::arg("reflectivity"), py::arg("velocity"),
             py::arg("elevation") = 0.0, py::arg("azimuth") = 0.0)
        .def("apply_boundary_conditions", &FluidSolver::apply_boundary_conditions)
        .def("enforce_incompressibility", &FluidSolver::enforce_incompressibility,
             py::arg("velocity"), py::arg("iterations") = 50)
        .def("sanitize_grid", &FluidSolver::sanitize_grid,
             py::arg("grid"), py::arg("fill_value") = 0.0,
             "Remove NaN/Inf values from grid, replace with fill_value")
        .def("sanitize_vector_field", &FluidSolver::sanitize_vector_field,
             py::arg("vf"), py::arg("fill_value") = 0.0,
             "Remove NaN/Inf values from vector field, replace with fill_value")
        .def("count_nan", &FluidSolver::count_nan,
             py::arg("grid"),
             "Count number of NaN/Inf values in grid")
        .def("advect_concentration", &FluidSolver::advect_concentration,
             py::arg("velocity"), py::arg("concentration"), 
             py::arg("concentration_new"), py::arg("dt"),
             "Advect concentration field using semi-Lagrangian method")
        .def("diffuse_concentration", &FluidSolver::diffuse_concentration,
             py::arg("concentration"), py::arg("concentration_new"),
             py::arg("dt"), py::arg("Kx"), py::arg("Ky"), py::arg("Kz"),
             "Diffuse concentration field with anisotropic diffusion")
        .def("apply_source_term", &FluidSolver::apply_source_term,
             py::arg("concentration"), py::arg("sources"),
             py::arg("current_time"), py::arg("dt"),
             "Apply aerosol source terms to concentration field")
        .def("apply_concentration_boundary_conditions", &FluidSolver::apply_concentration_boundary_conditions,
             py::arg("concentration"), py::arg("boundary_value") = 0.0,
             "Apply Dirichlet boundary conditions to concentration field")
        .def("solve_advection_diffusion", &FluidSolver::solve_advection_diffusion,
             py::arg("velocity"), py::arg("concentration"),
             py::arg("config"), py::arg("sources"), py::arg("output_dir") = "",
             "Solve full advection-diffusion equation for aerosol dispersion")
        .def("set_config", &FluidSolver::set_config,
             py::arg("config"),
             "Set solver configuration")
        .def("get_config", &FluidSolver::config,
             "Get solver configuration");

    m.def("get_omp_num_threads", []() { return omp_get_max_threads(); });
    m.def("set_omp_num_threads", [](int n) { omp_set_num_threads(n); });
}
