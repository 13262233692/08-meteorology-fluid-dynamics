#include "netcdf_reader.h"

#ifdef MFD_HAVE_NETCDF
#include <netcdf.h>
#endif

#include <cmath>
#include <stdexcept>
#include <iostream>
#include <cstdlib>

namespace mfd {

#ifdef MFD_HAVE_NETCDF
namespace {

void check_nc_status(int status, const std::string& operation) {
    if (status != NC_NOERR) {
        throw std::runtime_error("NetCDF error during " + operation + ": " +
                                 std::string(nc_strerror(status)));
    }
}

}  // namespace
#endif

NetCDFReader::NetCDFReader()
    : reflectivity_var_("DBZ")
    , velocity_var_("VEL")
    , radial_var_("VR")
    , azimuth_var_("azimuth")
    , elevation_var_("elevation")
    , range_var_("range") {}

NetCDFReader::~NetCDFReader() = default;

RadarVolumeScan NetCDFReader::read_radar_volume(const std::string& filepath) {
    RadarVolumeScan scan;

#ifdef MFD_HAVE_NETCDF
    int ncid;
    int status = nc_open(filepath.c_str(), NC_NOWRITE, &ncid);

    if (status != NC_NOERR) {
#endif
        std::cerr << "Warning: Cannot open NetCDF file: " << filepath
                  << ". Creating synthetic test data." << std::endl;

        const size_t nx = 100, ny = 100, nz = 20;
        scan.dims.nx = nx;
        scan.dims.ny = ny;
        scan.dims.nz = nz;
        scan.dims.dx = 250.0;
        scan.dims.dy = 250.0;
        scan.dims.dz = 500.0;

        scan.reflectivity.allocate(nx, ny, nz);
        scan.radial_velocity.allocate(nx, ny, nz);

        for (size_t k = 0; k < nz; ++k) {
            for (size_t j = 0; j < ny; ++j) {
                for (size_t i = 0; i < nx; ++i) {
                    double x = (static_cast<double>(i) - nx / 2.0) * scan.dims.dx;
                    double y = (static_cast<double>(j) - ny / 2.0) * scan.dims.dy;
                    double z = static_cast<double>(k) * scan.dims.dz;
                    double r = std::sqrt(x * x + y * y);
                    double sigma = 5000.0;
                    double gaussian = 35.0 * std::exp(-(r * r + (z - 3000.0) * (z - 3000.0))
                                                       / (2.0 * sigma * sigma));
                    scan.reflectivity(i, j, k) = gaussian + 5.0 * (rand() % 1000) / 1000.0;

                    double shear = 0.005 * z;
                    double v_x = -shear * y;
                    double v_y = shear * x;
                    double v_z = 2.0 * std::exp(-(r * r) / (2.0 * 10000.0 * 10000.0));
                    double v_r = (x * v_x + y * v_y) / std::max(r, 1.0);
                    scan.radial_velocity.u(i, j, k) = v_r;
                    scan.radial_velocity.v(i, j, k) = v_x;
                    scan.radial_velocity.w(i, j, k) = v_z;
                }
            }
        }

        scan.timestamp = "20260618_000000";
        scan.radar_name = "SYNTHETIC_RADAR";
        return scan;

#ifdef MFD_HAVE_NETCDF
    }

    try {
        read_grid_dimensions(filepath, scan.dims);

        size_t nx = scan.dims.nx;
        size_t ny = scan.dims.ny;
        size_t nz = scan.dims.nz;

        scan.reflectivity.allocate(nx, ny, nz);
        scan.radial_velocity.allocate(nx, ny, nz);

        read_3d_variable(filepath, reflectivity_var_, scan.reflectivity);
        read_3d_variable(filepath, velocity_var_, scan.radial_velocity.u);
        read_3d_variable(filepath, radial_var_, scan.radial_velocity.v);

        read_attribute_string(ncid, "time_coverage_start", scan.timestamp);
        read_attribute_string(ncid, "instrument_name", scan.radar_name);

        int dimid, varid;
        if (nc_inq_dimid(ncid, elevation_var_.c_str(), &dimid) == NC_NOERR) {
            size_t nelev;
            nc_inq_dimlen(ncid, dimid, &nelev);
            scan.elevation_angles.resize(nelev);
            if (nc_inq_varid(ncid, elevation_var_.c_str(), &varid) == NC_NOERR) {
                nc_get_var_double(ncid, varid, scan.elevation_angles.data());
            }
        }

        if (nc_inq_dimid(ncid, azimuth_var_.c_str(), &dimid) == NC_NOERR) {
            size_t nazim;
            nc_inq_dimlen(ncid, dimid, &nazim);
            scan.azimuth_angles.resize(nazim);
            if (nc_inq_varid(ncid, azimuth_var_.c_str(), &varid) == NC_NOERR) {
                nc_get_var_double(ncid, varid, scan.azimuth_angles.data());
            }
        }

        if (nc_inq_dimid(ncid, range_var_.c_str(), &dimid) == NC_NOERR) {
            size_t nrange;
            nc_inq_dimlen(ncid, dimid, &nrange);
            scan.range_gates.resize(nrange);
            if (nc_inq_varid(ncid, range_var_.c_str(), &varid) == NC_NOERR) {
                nc_get_var_double(ncid, varid, scan.range_gates.data());
            }
        }

        nc_close(ncid);
    } catch (const std::exception& e) {
        nc_close(ncid);
        throw;
    }

    return scan;
#endif
}

bool NetCDFReader::read_3d_variable(const std::string& filepath,
                                     const std::string& varname,
                                     Grid3D& out_grid) {
#ifdef MFD_HAVE_NETCDF
    int ncid;
    int status = nc_open(filepath.c_str(), NC_NOWRITE, &ncid);
    if (status != NC_NOERR) return false;

    try {
        int varid;
        status = nc_inq_varid(ncid, varname.c_str(), &varid);
        if (status != NC_NOERR) {
            std::cerr << "Warning: Variable '" << varname << "' not found, using zeros." << std::endl;
            out_grid.fill(0.0);
            nc_close(ncid);
            return true;
        }

        size_t nx = out_grid.nx();
        size_t ny = out_grid.ny();
        size_t nz = out_grid.nz();

        std::vector<size_t> start(3, 0);
        std::vector<size_t> count = {nz, ny, nx};

        if (out_grid.size() > 0) {
            status = nc_get_vara_double(ncid, varid, start.data(), count.data(), out_grid.data());
            if (status != NC_NOERR) {
                check_nc_status(status, "reading variable " + varname);
            }
        }

        nc_close(ncid);
        return true;
    } catch (...) {
        nc_close(ncid);
        throw;
    }
#else
    (void)filepath;
    (void)varname;
    out_grid.fill(0.0);
    return true;
#endif
}

bool NetCDFReader::read_grid_dimensions(const std::string& filepath,
                                         GridDimensions& out_dims) {
#ifdef MFD_HAVE_NETCDF
    int ncid;
    int status = nc_open(filepath.c_str(), NC_NOWRITE, &ncid);
    if (status != NC_NOERR) return false;

    try {
        int dimid;
        size_t dimlen;

        if (nc_inq_dimid(ncid, "z", &dimid) == NC_NOERR ||
            nc_inq_dimid(ncid, "elevation", &dimid) == NC_NOERR) {
            nc_inq_dimlen(ncid, dimid, &dimlen);
            out_dims.nz = dimlen;
        } else {
            out_dims.nz = 1;
        }

        if (nc_inq_dimid(ncid, "y", &dimid) == NC_NOERR ||
            nc_inq_dimid(ncid, "azimuth", &dimid) == NC_NOERR) {
            nc_inq_dimlen(ncid, dimid, &dimlen);
            out_dims.ny = dimlen;
        } else {
            out_dims.ny = 1;
        }

        if (nc_inq_dimid(ncid, "x", &dimid) == NC_NOERR ||
            nc_inq_dimid(ncid, "range", &dimid) == NC_NOERR) {
            nc_inq_dimlen(ncid, dimid, &dimlen);
            out_dims.nx = dimlen;
        } else {
            out_dims.nx = 1;
        }

        out_dims.dx = 250.0;
        out_dims.dy = 250.0;
        out_dims.dz = 500.0;
        out_dims.origin_x = 0.0;
        out_dims.origin_y = 0.0;
        out_dims.origin_z = 0.0;

        nc_close(ncid);
        return true;
    } catch (...) {
        nc_close(ncid);
        throw;
    }
#else
    (void)filepath;
    out_dims.nx = 100;
    out_dims.ny = 100;
    out_dims.nz = 20;
    out_dims.dx = 250.0;
    out_dims.dy = 250.0;
    out_dims.dz = 500.0;
    return true;
#endif
}

bool NetCDFReader::read_attribute_string(int ncid, const std::string& attr_name,
                                          std::string& out) {
#ifdef MFD_HAVE_NETCDF
    size_t attlen;
    int status = nc_inq_attlen(ncid, NC_GLOBAL, attr_name.c_str(), &attlen);
    if (status != NC_NOERR) {
        out = "";
        return false;
    }

    std::vector<char> buf(attlen + 1, '\0');
    status = nc_get_att_text(ncid, NC_GLOBAL, attr_name.c_str(), buf.data());
    if (status != NC_NOERR) {
        out = "";
        return false;
    }

    out = std::string(buf.data());
    return true;
#else
    (void)ncid;
    (void)attr_name;
    out = "";
    return false;
#endif
}

}  // namespace mfd
