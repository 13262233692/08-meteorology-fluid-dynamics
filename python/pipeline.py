#!/usr/bin/env python3
"""
Meteorology Fluid Dynamics - High Performance Computing Engine
顶层 Python 算法调度脚本
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'Release'))

try:
    import mfd
    print(f"[OK] mfd module loaded successfully")
except ImportError as e:
    print(f"[ERROR] Failed to import mfd module: {e}")
    print("Please build the project first using CMake")
    sys.exit(1)


class MeteorologyPipeline:
    """气象流体动力学计算流水线"""

    def __init__(self, num_threads=None, config=None):
        if num_threads is None:
            num_threads = os.cpu_count() or 4
        mfd.set_omp_num_threads(num_threads)
        print(f"[INFO] Using {mfd.get_omp_num_threads()} OpenMP threads")

        if config is None:
            config = mfd.SolverConfig()
            config.num_threads = num_threads
            config.time_step = 0.01
            config.max_iterations = 100
            config.convergence_tolerance = 1e-5

        self.solver = mfd.FluidSolver(config)
        self.reader = mfd.NetCDFReader()
        self.config = config

    def load_radar_data(self, filepath):
        """加载雷达体扫数据"""
        print(f"\n[INFO] Loading radar volume scan from: {filepath}")
        t0 = time.time()
        scan = self.reader.read_radar_volume(filepath)
        t1 = time.time()
        print(f"[INFO] Loaded data in {t1 - t0:.3f}s")
        print(f"       Radar: {scan.radar_name}")
        print(f"       Timestamp: {scan.timestamp}")
        print(f"       Grid dimensions: {scan.dims.nx} x {scan.dims.ny} x {scan.dims.nz}")
        print(f"       Grid spacing: dx={scan.dims.dx}m, dy={scan.dims.dy}m, dz={scan.dims.dz}m")
        return scan

    def reconstruct_3d_wind_field(self, scan):
        """从径向风速重建三维风场"""
        print("\n[INFO] Reconstructing 3D wind field from radial velocity...")
        t0 = time.time()

        velocity = mfd.VectorField3D()
        self.solver.compute_velocity_from_radial(
            scan.radial_velocity.u,
            scan.reflectivity,
            velocity,
            elevation=0.0,
            azimuth=0.0
        )

        t1 = time.time()
        print(f"[INFO] Wind field reconstruction completed in {t1 - t0:.3f}s")
        return velocity

    def solve_flow(self, velocity, reflectivity, iterations=-1):
        """求解简化的 Navier-Stokes 方程"""
        print("\n[INFO] Solving simplified Navier-Stokes equations...")
        t0 = time.time()

        self.solver.solve_navier_stokes_simplified(velocity, reflectivity, iterations)

        t1 = time.time()
        print(f"[INFO] Flow simulation completed in {t1 - t0:.3f}s")
        return velocity

    def compute_divergence(self, velocity):
        """计算风场散度"""
        print("\n[INFO] Computing divergence field...")
        t0 = time.time()

        divergence = mfd.Grid3D()
        self.solver.compute_divergence(velocity, divergence)

        t1 = time.time()
        print(f"[INFO] Divergence computed in {t1 - t0:.3f}s")

        div_np = divergence.to_numpy()
        print(f"       Min divergence: {div_np.min():.6f} s^-1")
        print(f"       Max divergence: {div_np.max():.6f} s^-1")
        print(f"       Mean divergence: {div_np.mean():.6f} s^-1")

        return divergence

    def compute_vorticity(self, velocity):
        """计算风场涡度"""
        print("\n[INFO] Computing vorticity field...")
        t0 = time.time()

        vorticity = mfd.VectorField3D()
        self.solver.compute_vorticity(velocity, vorticity)

        vorticity_mag = mfd.Grid3D()
        self.solver.compute_vorticity_magnitude(vorticity, vorticity_mag)

        t1 = time.time()
        print(f"[INFO] Vorticity computed in {t1 - t0:.3f}s")

        vm_np = vorticity_mag.to_numpy()
        print(f"       Min vorticity magnitude: {vm_np.min():.6f} s^-1")
        print(f"       Max vorticity magnitude: {vm_np.max():.6f} s^-1")
        print(f"       Mean vorticity magnitude: {vm_np.mean():.6f} s^-1")

        return vorticity, vorticity_mag

    def process_file(self, filepath, ns_iterations=50):
        """完整的文件处理流水线"""
        total_t0 = time.time()

        print("=" * 60)
        print("  Meteorology Fluid Dynamics Processing Pipeline")
        print("=" * 60)

        scan = self.load_radar_data(filepath)
        velocity = self.reconstruct_3d_wind_field(scan)
        velocity = self.solve_flow(velocity, scan.reflectivity, ns_iterations)

        divergence = self.compute_divergence(velocity)
        vorticity, vorticity_mag = self.compute_vorticity(velocity)

        u_np, v_np, w_np = velocity.to_numpy_tuple()
        div_np = divergence.to_numpy()
        vort_mag_np = vorticity_mag.to_numpy()
        refl_np = scan.reflectivity.to_numpy()

        results = {
            'velocity_u': u_np,
            'velocity_v': v_np,
            'velocity_w': w_np,
            'divergence': div_np,
            'vorticity_u': vorticity.u.to_numpy(),
            'vorticity_v': vorticity.v.to_numpy(),
            'vorticity_w': vorticity.w.to_numpy(),
            'vorticity_magnitude': vort_mag_np,
            'reflectivity': refl_np,
            'dimensions': {
                'nx': scan.dims.nx,
                'ny': scan.dims.ny,
                'nz': scan.dims.nz,
                'dx': scan.dims.dx,
                'dy': scan.dims.dy,
                'dz': scan.dims.dz,
            },
            'radar_name': scan.radar_name,
            'timestamp': scan.timestamp,
        }

        total_t1 = time.time()
        print(f"\n{'=' * 60}")
        print(f"  Total processing time: {total_t1 - total_t0:.3f}s")
        print(f"{'=' * 60}")

        return results

    def analyze_results(self, results):
        """对计算结果进行诊断分析"""
        print("\n" + "=" * 60)
        print("  Diagnostic Analysis")
        print("=" * 60)

        div = results['divergence']
        vort_mag = results['vorticity_magnitude']
        refl = results['reflectivity']

        high_conv_mask = div < -1e-4
        high_div_mask = div > 1e-4
        high_vort_mask = vort_mag > 1e-3
        strong_refl_mask = refl > 35.0

        print(f"\n  Points with strong convergence (div < -1e-4): {high_conv_mask.sum()}")
        print(f"  Points with strong divergence (div > 1e-4): {high_div_mask.sum()}")
        print(f"  Points with strong vorticity (|vort| > 1e-3): {high_vort_mask.sum()}")
        print(f"  Points with strong reflectivity (DBZ > 35): {strong_refl_mask.sum()}")

        severe_mask = high_conv_mask & high_vort_mask & strong_refl_mask
        print(f"\n  *** Severe weather signature points: {severe_mask.sum()} ***")

        if severe_mask.sum() > 0:
            severe_coords = np.argwhere(severe_mask)
            print(f"      Clusters detected at levels: {np.unique(severe_coords[:, 0])}")

        return {
            'severe_points': severe_mask.sum(),
            'high_convergence': high_conv_mask.sum(),
            'high_vorticity': high_vort_mask.sum(),
        }


def benchmark(grid_size=64, ns_iterations=20):
    """性能基准测试"""
    print("\n" + "=" * 60)
    print("  Performance Benchmark")
    print("=" * 60)

    pipeline = MeteorologyPipeline()

    print(f"\n[INFO] Creating synthetic grid: {grid_size}^3")
    nx = ny = nz = grid_size
    velocity = mfd.VectorField3D(nx, ny, nz, 250.0, 250.0, 500.0)
    reflectivity = mfd.Grid3D(nx, ny, nz, 250.0, 250.0, 500.0)

    u_np = velocity.u.to_numpy()
    v_np = velocity.v.to_numpy()
    w_np = velocity.w.to_numpy()
    r_np = reflectivity.to_numpy()

    cx, cy, cz = nx / 2.0, ny / 2.0, nz / 2.0
    sigma = nx * 250.0 / 4.0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                x = (i - cx) * 250.0
                y = (j - cy) * 250.0
                z = (k - cz) * 500.0
                r2 = x * x + y * y
                gauss_v = np.exp(-r2 / (2 * sigma * sigma))
                gauss_r = np.exp(-(r2 + z * z) / (2 * sigma * sigma))
                u_np[k, j, i] = -y * 0.01 * gauss_v
                v_np[k, j, i] = x * 0.01 * gauss_v
                w_np[k, j, i] = 2.0 * gauss_v
                r_np[k, j, i] = 35.0 * gauss_r

    print("\n[INFO] Benchmarking Navier-Stokes solver...")
    t0 = time.time()
    pipeline.solver.solve_navier_stokes_simplified(velocity, reflectivity, ns_iterations)
    t1 = time.time()

    elapsed = t1 - t0
    grid_points = grid_size ** 3
    points_per_sec = grid_points * ns_iterations / elapsed
    print(f"\n[RESULTS]")
    print(f"  Grid size: {grid_size}^3 = {grid_points:,} points")
    print(f"  NS iterations: {ns_iterations}")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Throughput: {points_per_sec:,.0f} grid-points/second")
    print(f"  Per iteration: {elapsed/ns_iterations*1000:.2f} ms")

    print("\n[INFO] Benchmarking divergence computation...")
    divergence = mfd.Grid3D()
    t0 = time.time()
    pipeline.solver.compute_divergence(velocity, divergence)
    t1 = time.time()
    print(f"  Divergence time: {(t1-t0)*1000:.2f} ms")

    print("\n[INFO] Benchmarking vorticity computation...")
    vorticity = mfd.VectorField3D()
    t0 = time.time()
    pipeline.solver.compute_vorticity(velocity, vorticity)
    vorticity_mag = mfd.Grid3D()
    pipeline.solver.compute_vorticity_magnitude(vorticity, vorticity_mag)
    t1 = time.time()
    print(f"  Vorticity time: {(t1-t0)*1000:.2f} ms")

    return elapsed


def main():
    print("Meteorology Fluid Dynamics HPC Engine")
    print(f"NumPy version: {np.__version__}")
    print(f"mfd version: {mfd.__doc__}")

    if len(sys.argv) > 1 and sys.argv[1] == '--benchmark':
        grid_size = int(sys.argv[2]) if len(sys.argv) > 2 else 64
        benchmark(grid_size=grid_size)
        return

    pipeline = MeteorologyPipeline(num_threads=min(8, os.cpu_count() or 4))

    results = pipeline.process_file("synthetic_radar_data.nc", ns_iterations=50)

    analysis = pipeline.analyze_results(results)

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "analysis_results.npz")

    np.savez(output_path,
             velocity_u=results['velocity_u'],
             velocity_v=results['velocity_v'],
             velocity_w=results['velocity_w'],
             divergence=results['divergence'],
             vorticity_magnitude=results['vorticity_magnitude'],
             reflectivity=results['reflectivity'])
    print(f"\n[INFO] Results saved to: {output_path}")

    print("\n" + "=" * 60)
    print("  Processing Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
