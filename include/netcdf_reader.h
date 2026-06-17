#ifndef MFD_NETCDF_READER_H
#define MFD_NETCDF_READER_H

#include "grid3d.h"
#include <string>
#include <vector>

namespace mfd {

struct RadarVolumeScan {
    Grid3D reflectivity;
    VectorField3D radial_velocity;
    GridDimensions dims;
    std::vector<double> elevation_angles;
    std::vector<double> azimuth_angles;
    std::vector<double> range_gates;
    std::string timestamp;
    std::string radar_name;
};

class NetCDFReader {
public:
    NetCDFReader();
    ~NetCDFReader();

    RadarVolumeScan read_radar_volume(const std::string& filepath);

    bool read_3d_variable(const std::string& filepath,
                          const std::string& varname,
                          Grid3D& out_grid);

    bool read_grid_dimensions(const std::string& filepath,
                              GridDimensions& out_dims);

    void set_reflectivity_varname(const std::string& name) { reflectivity_var_ = name; }
    void set_velocity_varname(const std::string& name) { velocity_var_ = name; }
    void set_radial_varname(const std::string& name) { radial_var_ = name; }

    const std::string& reflectivity_varname() const { return reflectivity_var_; }
    const std::string& velocity_varname() const { return velocity_var_; }
    const std::string& radial_varname() const { return radial_var_; }

private:
    std::string reflectivity_var_;
    std::string velocity_var_;
    std::string radial_var_;
    std::string azimuth_var_;
    std::string elevation_var_;
    std::string range_var_;

    bool read_attribute_string(int ncid, const std::string& attr_name, std::string& out);
};

}  // namespace mfd

#endif  // MFD_NETCDF_READER_H
