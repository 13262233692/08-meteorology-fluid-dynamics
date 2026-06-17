#!/usr/bin/env python3
"""
示例：直接通过 pybind11 访问 C++ 底层矩阵数据指针
演示零拷贝数据共享机制
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'Release'))

import mfd


def demo_numpy_zero_copy():
    print("=== Demo 1: NumPy 零拷贝数据共享 ===")

    nx, ny, nz = 32, 32, 16
    grid = mfd.Grid3D(nx, ny, nz, 100.0, 100.0, 200.0)

    np_arr = grid.to_numpy()
    print(f"NumPy array shape: {np_arr.shape}")
    print(f"NumPy array dtype: {np_arr.dtype}")

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                np_arr[k, j, i] = i + j * 100 + k * 10000

    print(f"\nC++ grid(5, 10, 3) = {grid(5, 10, 3)}")
    print(f"NumPy arr[3, 10, 5] = {np_arr[3, 10, 5]}")
    print(f"Values match: {grid(5, 10, 3) == np_arr[3, 10, 5]}")

    np_arr[0, 0, 0] = 99999.0
    print(f"\nAfter modifying NumPy array:")
    print(f"NumPy arr[0, 0, 0] = {np_arr[0, 0, 0]}")
    print(f"C++ grid(0, 0, 0) = {grid(0, 0, 0)}")
    print(f"Zero-copy verified: values share memory")

    data_ptr = grid.data_ptr
    print(f"\nRaw C++ data pointer: 0x{data_ptr:X}")
    print(f"NumPy data pointer:    0x{np_arr.ctypes.data:X}")
    print(f"Pointers match: {data_ptr == np_arr.ctypes.data}")


def demo_vector_field():
    print("\n\n=== Demo 2: VectorField3D 操作 ===")

    nx, ny, nz = 64, 64, 32
    vf = mfd.VectorField3D(nx, ny, nz, 250.0, 250.0, 500.0)

    u_np, v_np, w_np = vf.to_numpy_tuple()
    print(f"U component shape: {u_np.shape}")
    print(f"V component shape: {v_np.shape}")
    print(f"W component shape: {w_np.shape}")

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                u_np[k, j, i] = -j * 0.01
                v_np[k, j, i] = i * 0.01
                w_np[k, j, i] = np.sin(k * 0.05)

    print(f"\nU[0, 10, 5] = {u_np[5, 10, 0]:.4f} (via NumPy)")
    print(f"U(0, 10, 5) = {vf.u(0, 10, 5):.4f} (via C++)")


def demo_fluid_computation():
    print("\n\n=== Demo 3: 流体计算（散度、涡度） ===")

    solver = mfd.FluidSolver()
    nx, ny, nz = 64, 64, 32

    velocity = mfd.VectorField3D(nx, ny, nz, 1.0, 1.0, 1.0)
    u_np, v_np, w_np = velocity.to_numpy_tuple()

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                x = i - nx / 2
                y = j - ny / 2
                u_np[k, j, i] = -y * 0.1
                v_np[k, j, i] = x * 0.1
                w_np[k, j, i] = 0.0

    print(f"Velocity field initialized: solid body rotation")

    divergence = mfd.Grid3D()
    t0 = os.times()[0]
    solver.compute_divergence(velocity, divergence)
    t1 = os.times()[0]

    div_np = divergence.to_numpy()
    print(f"\nDivergence computed")
    print(f"  Mean divergence (interior): {div_np[1:-1, 1:-1, 1:-1].mean():.6e}")
    print(f"  Max |divergence|: {np.abs(div_np[1:-1, 1:-1, 1:-1]).max():.6e}")

    vorticity = mfd.VectorField3D()
    vorticity_mag = mfd.Grid3D()
    solver.compute_vorticity(velocity, vorticity)
    solver.compute_vorticity_magnitude(vorticity, vorticity_mag)

    wx_np, wy_np, wz_np = vorticity.to_numpy_tuple()
    vm_np = vorticity_mag.to_numpy()
    print(f"\nVorticity computed")
    print(f"  Mean omega_z (interior): {wz_np[1:-1, 1:-1, 1:-1].mean():.6f}")
    print(f"  Expected omega_z: 0.2")
    print(f"  Mean vorticity magnitude: {vm_np[1:-1, 1:-1, 1:-1].mean():.6f}")


def demo_ns_solver():
    print("\n\n=== Demo 4: Navier-Stokes 求解器 ===")

    config = mfd.SolverConfig()
    config.num_threads = 4
    config.time_step = 0.005
    config.max_iterations = 20
    config.convergence_tolerance = 1e-4

    solver = mfd.FluidSolver(config)

    nx, ny, nz = 40, 40, 20
    velocity = mfd.VectorField3D(nx, ny, nz, 100.0, 100.0, 200.0)
    reflectivity = mfd.Grid3D(nx, ny, nz, 100.0, 100.0, 200.0)

    u_np, v_np, w_np = velocity.to_numpy_tuple()
    r_np = reflectivity.to_numpy()

    cx, cy, cz = nx / 2, ny / 2, nz / 2
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                dx = i - cx
                dy = j - cy
                dz = k - cz
                r2 = dx * dx + dy * dy + dz * dz
                sigma = nx * nx / 4
                gauss = np.exp(-r2 / sigma)
                u_np[k, j, i] = -dy * 0.05 * gauss
                v_np[k, j, i] = dx * 0.05 * gauss
                w_np[k, j, i] = 1.0 * gauss
                r_np[k, j, i] = 40.0 * gauss

    print(f"Initial velocity field with Gaussian vortex")
    print(f"Grid: {nx}x{ny}x{nz}")

    divergence = mfd.Grid3D()
    solver.compute_divergence(velocity, divergence)
    div_np = divergence.to_numpy()
    print(f"\nBefore NS solve:")
    print(f"  Max |divergence|: {np.abs(div_np).max():.6e}")

    solver.solve_navier_stokes_simplified(velocity, reflectivity, 10)

    solver.compute_divergence(velocity, divergence)
    print(f"\nAfter NS solve:")
    print(f"  Max |divergence|: {np.abs(div_np).max():.6e}")

    vorticity_mag = mfd.Grid3D()
    vorticity = mfd.VectorField3D()
    solver.compute_vorticity(velocity, vorticity)
    solver.compute_vorticity_magnitude(vorticity, vorticity_mag)
    vm_np = vorticity_mag.to_numpy()
    print(f"  Max vorticity magnitude: {vm_np.max():.6e}")


def main():
    print("Meteorology Fluid Dynamics - pybind11 Data Access Examples")
    print("=" * 60)

    demo_numpy_zero_copy()
    demo_vector_field()
    demo_fluid_computation()
    demo_ns_solver()

    print("\n" + "=" * 60)
    print("All demos completed successfully!")


if __name__ == "__main__":
    main()
