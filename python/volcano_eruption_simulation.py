#!/usr/bin/env python3
"""Volcano Eruption Aerosol Dispersion Simulation with HDF5 Output"""
import sys
import os
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'Release'))
import mfd

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("[WARNING] h5py not available, HDF5 output will be disabled")
    print("          Install with: pip install h5py")

def create_steady_wind_field(nx=80, ny=80, nz=40, scenario='typhoon'):
    """Create a steady-state 3D wind field for dispersion simulation"""
    print(f"\n[INFO] Creating steady wind field ({scenario} scenario)...")

    velocity = mfd.VectorField3D(nx, ny, nz, 1000.0, 1000.0, 500.0)
    reflectivity = mfd.Grid3D(nx, ny, nz, 1000.0, 1000.0, 500.0)

    u_np = velocity.u.to_numpy()
    v_np = velocity.v.to_numpy()
    w_np = velocity.w.to_numpy()
    r_np = reflectivity.to_numpy()

    cx, cy, cz = nx / 2.0, ny / 2.0, nz / 2.0

    if scenario == 'typhoon':
        eye_radius = 10000.0
        eye_wall_radius = 30000.0

        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    x = (i - cx) * 1000.0
                    y = (j - cy) * 1000.0
                    z = k * 500.0

                    r2 = x * x + y * y
                    r = np.sqrt(r2)

                    if r < eye_radius:
                        u_np[k, j, i] = 0.0
                        v_np[k, j, i] = 0.0
                        w_np[k, j, i] = -2.0
                        r_np[k, j, i] = 10.0
                    elif r < eye_wall_radius:
                        angle = np.arctan2(y, x)
                        speed = 60.0 * np.exp(-(r - eye_radius)**2 / (2 * 10000.0**2))
                        u_np[k, j, i] = -np.sin(angle) * speed
                        v_np[k, j, i] = np.cos(angle) * speed
                        w_np[k, j, i] = 8.0 * np.exp(-(r - eye_radius)**2 / (2 * 5000.0**2))
                        r_np[k, j, i] = 45.0 * np.exp(-(r - eye_radius)**2 / (2 * 8000.0**2))
                    else:
                        angle = np.arctan2(y, x)
                        speed = 20.0 * (eye_wall_radius / r)
                        u_np[k, j, i] = -np.sin(angle) * speed
                        v_np[k, j, i] = np.cos(angle) * speed
                        w_np[k, j, i] = 1.0 * np.exp(-r2 / (2 * 50000.0**2))
                        r_np[k, j, i] = 20.0 * np.exp(-r2 / (2 * 60000.0**2))

                    w_np[k, j, i] *= np.exp(-(z - 10000.0)**2 / (2 * 15000.0**2))

    elif scenario == 'uniform':
        wind_speed = 15.0
        wind_direction = 45.0
        angle_rad = np.radians(wind_direction)

        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    z = k * 500.0
                    speed_factor = 1.0 + 0.5 * np.tanh((z - 5000.0) / 2000.0)
                    u_np[k, j, i] = wind_speed * np.cos(angle_rad) * speed_factor
                    v_np[k, j, i] = wind_speed * np.sin(angle_rad) * speed_factor
                    w_np[k, j, i] = 0.5 * speed_factor
                    r_np[k, j, i] = 0.0

    elif scenario == 'mountain':
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    x = (i - cx) * 1000.0
                    y = (j - cy) * 1000.0
                    z = k * 500.0

                    u_np[k, j, i] = 20.0 + 5.0 * np.sin(y / 20000.0)
                    v_np[k, j, i] = 0.0
                    r_np[k, j, i] = 0.0

                    mountain_height = 3000.0 * np.exp(-(x**2 + y**2) / (2 * 15000.0**2))
                    if z < mountain_height + 1000.0:
                        w_np[k, j, i] = 10.0 * (1.0 - z / (mountain_height + 1000.0))
                    else:
                        w_np[k, j, i] = 2.0 * np.exp(-(z - mountain_height)**2 / (2 * 5000.0**2))

    print(f"  Wind field created: {nx}x{ny}x{nz}")
    print(f"  Max |u|={np.max(np.abs(u_np)):.2f} m/s, |v|={np.max(np.abs(v_np)):.2f} m/s, |w|={np.max(np.abs(w_np)):.2f} m/s")

    return velocity, reflectivity

def setup_volcano_source(nx=80, ny=80, nz=40, grid_dx=1000.0, grid_dy=1000.0, grid_dz=500.0):
    """Setup volcano eruption source"""
    source = mfd.AerosolSource()

    source.i = nx // 2 - 15
    source.j = ny // 2
    source.k = 2

    source.name = "Mount Pinatubo Eruption"
    source.strength = 5.0e7
    source.radius = 3000.0
    source.start_time = 0.0
    source.end_time = 3 * 3600.0

    source_lat = 15.13
    source_lon = 120.35
    source_alt = 1486.0

    print(f"\n[INFO] Volcano source configured:")
    print(f"  Name: {source.name}")
    print(f"  Grid position: ({source.i}, {source.j}, {source.k})")
    print(f"  Geographic: {source_lat}°N, {source_lon}°E, {source_alt}m")
    print(f"  Strength: {source.strength:.2e} particles/s")
    print(f"  Radius: {source.radius:.0f} m")
    print(f"  Duration: {source.start_time/3600:.1f}h - {source.end_time/3600:.1f}h")

    return source, (source_lat, source_lon, source_alt)

def save_frame_hdf5(concentration, time_sec, frame_index, output_dir, 
                     grid_dims, sources, sim_config, geo_info):
    """Save simulation frame as HDF5"""
    if not HAS_H5PY:
        return None

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime(2025, 6, 1, 0, 0, 0) + timedelta(seconds=time_sec)
    filename = os.path.join(output_dir, f"aerosol_frame_{frame_index:04d}_{timestamp.strftime('%Y%m%d_%H%M%S')}.h5")

    conc_np = concentration.to_numpy()

    with h5py.File(filename, 'w') as f:
        time_grp = f.create_group('time')
        time_grp.create_dataset('simulation_time_seconds', data=time_sec)
        time_grp.create_dataset('simulation_time_hours', data=time_sec / 3600.0)
        time_grp.create_dataset('timestamp_utc', data=timestamp.isoformat())
        time_grp.create_dataset('frame_index', data=frame_index)

        grid_grp = f.create_group('grid')
        grid_grp.create_dataset('dimensions', data=[grid_dims[0], grid_dims[1], grid_dims[2]])
        grid_grp.create_dataset('spacing', data=[grid_dims[3], grid_dims[4], grid_dims[5]])
        grid_grp.create_dataset('origin', data=[grid_dims[6], grid_dims[7], grid_dims[8]])
        grid_grp.create_dataset('units', data='meters')

        data_grp = f.create_group('data')
        dset = data_grp.create_dataset('aerosol_concentration', 
                                       data=conc_np,
                                       compression='gzip',
                                       compression_opts=4)
        dset.attrs['units'] = 'particles/m³'
        dset.attrs['long_name'] = 'Atmospheric Aerosol Number Concentration'
        dset.attrs['valid_min'] = 0.0
        dset.attrs['valid_max'] = np.max(conc_np)

        stats_grp = f.create_group('statistics')
        stats_grp.create_dataset('min_concentration', data=np.min(conc_np))
        stats_grp.create_dataset('max_concentration', data=np.max(conc_np))
        stats_grp.create_dataset('mean_concentration', data=np.mean(conc_np))
        stats_grp.create_dataset('std_concentration', data=np.std(conc_np))
        stats_grp.create_dataset('total_particles', 
                                data=np.sum(conc_np) * grid_dims[3] * grid_dims[4] * grid_dims[5])
        stats_grp.create_dataset('non_zero_cells', data=np.sum(conc_np > 1e-10))

        sources_grp = f.create_group('sources')
        for idx, src in enumerate(sources):
            src_grp = sources_grp.create_group(f'source_{idx:02d}')
            src_grp.create_dataset('name', data=src.name)
            src_grp.create_dataset('grid_position', data=[src.i, src.j, src.k])
            src_grp.create_dataset('strength', data=src.strength)
            src_grp.create_dataset('radius_meters', data=src.radius)
            src_grp.create_dataset('active_time', data=[src.start_time, src.end_time])

        if geo_info:
            geo_grp = f.create_group('geolocation')
            geo_grp.create_dataset('source_latitude', data=geo_info[0])
            geo_grp.create_dataset('source_longitude', data=geo_info[1])
            geo_grp.create_dataset('source_altitude', data=geo_info[2])
            geo_grp.create_dataset('crs', data='EPSG:4326')

        config_grp = f.create_group('config')
        config_grp.create_dataset('diffusion_Kx', data=sim_config.diffusion_coeff_x)
        config_grp.create_dataset('diffusion_Ky', data=sim_config.diffusion_coeff_y)
        config_grp.create_dataset('diffusion_Kz', data=sim_config.diffusion_coeff_z)
        config_grp.create_dataset('time_step', data=sim_config.time_step)
        config_grp.create_dataset('background_concentration', data=sim_config.background_concentration)

        f.attrs['title'] = 'Volcanic Aerosol Dispersion Simulation'
        f.attrs['institution'] = 'Meteorology Fluid Dynamics Lab'
        f.attrs['source'] = 'Numerical Simulation'
        f.attrs['history'] = f'Created on {datetime.utcnow().isoformat()} UTC'
        f.attrs['conventions'] = 'CF-1.8'

    return filename

def run_volcano_simulation(nx=80, ny=80, nz=40, scenario='typhoon', 
                           output_dir='simulation_output',
                           simulate_hours=12, output_interval_hours=1):
    """Run complete volcano eruption simulation"""
    print("=" * 70)
    print("  VOLCANIC AEROSOL DISPERSION SIMULATION")
    print("=" * 70)
    print(f"  Grid: {nx} x {ny} x {nz}")
    print(f"  Scenario: {scenario}")
    print(f"  Duration: {simulate_hours} hours")
    print(f"  Output interval: {output_interval_hours} hours")
    print(f"  HDF5 output: {'ENABLED' if HAS_H5PY else 'DISABLED'}")
    print("=" * 70)

    solver = mfd.FluidSolver()

    velocity, reflectivity = create_steady_wind_field(nx, ny, nz, scenario)

    ns_config = mfd.SolverConfig()
    ns_config.time_step = 0.02
    ns_config.max_iterations = 30
    ns_config.num_threads = 8
    solver.set_config(ns_config)

    print("\n[INFO] Preprocessing wind field with Navier-Stokes solver...")
    solver.solve_navier_stokes_simplified(velocity, reflectivity, 25)

    source, geo_info = setup_volcano_source(nx, ny, nz, 1000.0, 1000.0, 500.0)
    sources = [source]

    ad_config = mfd.AdvectionDiffusionConfig()
    ad_config.diffusion_coeff_x = 200.0
    ad_config.diffusion_coeff_y = 200.0
    ad_config.diffusion_coeff_z = 20.0
    ad_config.time_step = 30.0
    ad_config.total_time = simulate_hours * 3600.0
    ad_config.output_interval = output_interval_hours * 3600.0
    ad_config.background_concentration = 0.0
    ad_config.num_threads = 8

    concentration = mfd.Grid3D(nx, ny, nz, 1000.0, 1000.0, 500.0)

    grid_dims = (nx, ny, nz, 1000.0, 1000.0, 500.0, 0.0, 0.0, 0.0)

    print("\n" + "=" * 70)
    print("  RUNNING ADVECTION-DIFFUSION SIMULATION")
    print("=" * 70)

    current_time = 0.0
    next_output_time = 0.0
    frame_count = 0
    saved_files = []

    conc_prev = mfd.Grid3D(nx, ny, nz, 1000.0, 1000.0, 500.0)
    conc_temp = mfd.Grid3D(nx, ny, nz, 1000.0, 1000.0, 500.0)

    concentration.fill(0.0)
    solver.apply_concentration_boundary_conditions(concentration, 0.0)

    total_steps = int(np.ceil(ad_config.total_time / ad_config.time_step))
    steps_per_output = int(ad_config.output_interval / ad_config.time_step)
    print(f"  Total time steps: {total_steps}")
    print(f"  Steps per output: {steps_per_output}")
    print(f"  Output frames: {total_steps // steps_per_output + 1}")
    print()

    step = 0
    while current_time < ad_config.total_time:
        dt = min(ad_config.time_step, ad_config.total_time - current_time)

        conc_prev.swap(concentration)
        solver.sanitize_grid(conc_prev, 0.0)

        solver.advect_concentration(velocity, conc_prev, conc_temp, dt)
        solver.apply_source_term(conc_temp, sources, current_time, dt)
        solver.diffuse_concentration(conc_temp, concentration, dt,
                                      ad_config.diffusion_coeff_x,
                                      ad_config.diffusion_coeff_y,
                                      ad_config.diffusion_coeff_z)

        solver.sanitize_grid(concentration, 0.0)
        solver.apply_concentration_boundary_conditions(concentration, 0.0)

        current_time += dt
        step += 1

        if current_time >= next_output_time - 1e-10:
            conc_np = concentration.to_numpy()
            max_c = np.max(conc_np)
            total_mass = np.sum(conc_np) * 1000.0 * 1000.0 * 500.0
            nans = solver.count_nan(concentration)

            hours = int(current_time / 3600)
            minutes = int((current_time - hours * 3600) / 60)

            print(f"  [T+{hours:02d}:{minutes:02d}] Step {step}/{total_steps} | "
                  f"max_C={max_c:.2e} | total_mass={total_mass:.2e} | NaN={nans}")

            if HAS_H5PY:
                fname = save_frame_hdf5(concentration, current_time, frame_count,
                                        output_dir, grid_dims, sources, 
                                        ad_config, geo_info)
                saved_files.append(fname)
                if fname:
                    fsize = os.path.getsize(fname) / (1024 * 1024)
                    print(f"    Saved: {os.path.basename(fname)} ({fsize:.1f} MB)")

            next_output_time += ad_config.output_interval
            frame_count += 1

    print("\n" + "=" * 70)
    print("  SIMULATION COMPLETE")
    print("=" * 70)
    print(f"  Total frames saved: {frame_count}")
    print(f"  Output directory: {os.path.abspath(output_dir)}")

    if saved_files:
        total_size = sum(os.path.getsize(f) for f in saved_files) / (1024 * 1024)
        print(f"  Total HDF5 size: {total_size:.1f} MB")
        print("\n  Saved files:")
        for f in saved_files[:5]:
            print(f"    - {os.path.basename(f)}")
        if len(saved_files) > 5:
            print(f"    ... and {len(saved_files) - 5} more")

    print("=" * 70)

    return concentration, saved_files

def analyze_hdf5_output(output_dir):
    """Analyze HDF5 output files and generate summary"""
    if not HAS_H5PY:
        print("[INFO] h5py not available, skipping HDF5 analysis")
        return

    import glob
    h5_files = sorted(glob.glob(os.path.join(output_dir, 'aerosol_frame_*.h5')))

    if not h5_files:
        print("[INFO] No HDF5 files found")
        return

    print("\n" + "=" * 70)
    print("  HDF5 OUTPUT ANALYSIS")
    print("=" * 70)

    times = []
    max_concentrations = []
    total_masses = []

    for f in h5_files:
        with h5py.File(f, 'r') as hf:
            t = hf['time/simulation_time_hours'][()]
            max_c = hf['statistics/max_concentration'][()]
            mass = hf['statistics/total_particles'][()]
            times.append(t)
            max_concentrations.append(max_c)
            total_masses.append(mass)
            print(f"  T+{t:5.1f}h: max_C={max_c:.2e}, total_mass={mass:.2e}")

    print("\n  Summary:")
    print(f"    Peak concentration: {max(max_concentrations):.2e} particles/m³")
    print(f"    Final total mass: {total_masses[-1]:.2e} particles")
    print(f"    Mass conservation: {total_masses[-1]/total_masses[0]*100:.1f}% of initial")
    print("=" * 70)

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Volcanic Aerosol Dispersion Simulation')
    parser.add_argument('--nx', type=int, default=60, help='Grid X dimension')
    parser.add_argument('--ny', type=int, default=60, help='Grid Y dimension')
    parser.add_argument('--nz', type=int, default=30, help='Grid Z dimension')
    parser.add_argument('--scenario', type=str, default='typhoon',
                       choices=['typhoon', 'uniform', 'mountain'],
                       help='Wind field scenario')
    parser.add_argument('--hours', type=float, default=6.0, help='Simulation hours')
    parser.add_argument('--output-interval', type=float, default=0.5,
                       help='Output interval in hours')
    parser.add_argument('--output-dir', type=str, default='volcano_simulation_output',
                       help='Output directory')
    parser.add_argument('--no-hdf5', action='store_true', help='Disable HDF5 output')

    args = parser.parse_args()

    if args.no_hdf5:
        global HAS_H5PY
        HAS_H5PY = False

    concentration, saved_files = run_volcano_simulation(
        nx=args.nx, ny=args.ny, nz=args.nz,
        scenario=args.scenario,
        output_dir=args.output_dir,
        simulate_hours=args.hours,
        output_interval_hours=args.output_interval
    )

    if HAS_H5PY and saved_files:
        analyze_hdf5_output(args.output_dir)

    print("\n[DONE] Simulation complete!")

if __name__ == "__main__":
    main()
