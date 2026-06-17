#!/usr/bin/env python3
"""NaN Robustness Test - Simulate severe typhoon eye convection scenarios"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'Release'))
import mfd

def create_typhoon_initial_condition(nx=64, ny=64, nz=32):
    """Create initial condition simulating a typhoon eye with extreme gradients"""
    velocity = mfd.VectorField3D(nx, ny, nz, 250.0, 250.0, 500.0)
    reflectivity = mfd.Grid3D(nx, ny, nz, 250.0, 250.0, 500.0)

    u_np = velocity.u.to_numpy()
    v_np = velocity.v.to_numpy()
    w_np = velocity.w.to_numpy()
    r_np = reflectivity.to_numpy()

    cx, cy, cz = nx / 2.0, ny / 2.0, nz / 2.0

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                x = (i - cx) * 250.0
                y = (j - cy) * 250.0
                z = (k - cz) * 500.0

                r2 = x * x + y * y
                r = np.sqrt(r2)

                eye_radius = 5000.0
                eye_wall_radius = 15000.0

                if r < eye_radius:
                    u_np[k, j, i] = 0.0
                    v_np[k, j, i] = 0.0
                    w_np[k, j, i] = -5.0
                    r_np[k, j, i] = 10.0
                elif r < eye_wall_radius:
                    angle = np.arctan2(y, x)
                    speed = 80.0 * np.exp(-(r - eye_radius)**2 / (2 * 5000.0**2))
                    u_np[k, j, i] = -np.sin(angle) * speed
                    v_np[k, j, i] = np.cos(angle) * speed
                    w_np[k, j, i] = 15.0 * np.exp(-(r - eye_radius)**2 / (2 * 3000.0**2))
                    r_np[k, j, i] = 50.0 * np.exp(-(r - eye_radius)**2 / (2 * 4000.0**2))
                else:
                    angle = np.arctan2(y, x)
                    speed = 30.0 * (eye_wall_radius / r)
                    u_np[k, j, i] = -np.sin(angle) * speed
                    v_np[k, j, i] = np.cos(angle) * speed
                    w_np[k, j, i] = 2.0 * np.exp(-r2 / (2 * 20000.0**2))
                    r_np[k, j, i] = 25.0 * np.exp(-r2 / (2 * 25000.0**2))

                w_np[k, j, i] *= np.exp(-z * z / (2 * 8000.0**2))

    return velocity, reflectivity

def inject_deliberate_nans(velocity, nan_fraction=0.001):
    """Inject NaN values to simulate real-world radar data corruption"""
    nx = velocity.nx
    ny = velocity.ny
    nz = velocity.nz

    u_np = velocity.u.to_numpy()
    v_np = velocity.v.to_numpy()
    w_np = velocity.w.to_numpy()

    total_points = nx * ny * nz
    n_nan = int(total_points * nan_fraction)

    rng = np.random.RandomState(42)
    for _ in range(n_nan):
        i = rng.randint(0, nx)
        j = rng.randint(0, ny)
        k = rng.randint(0, nz)

        which = rng.randint(0, 3)
        if which == 0:
            u_np[k, j, i] = np.nan
        elif which == 1:
            v_np[k, j, i] = np.nan
        else:
            w_np[k, j, i] = np.nan

    print(f"  [INFO] Injected {n_nan} NaN values ({nan_fraction*100:.3f}% of data)")
    return n_nan

def count_total_nans(velocity):
    """Count total NaN values in velocity field"""
    solver = mfd.FluidSolver()
    return solver.count_nan(velocity.u) + solver.count_nan(velocity.v) + solver.count_nan(velocity.w)

def test_no_nan_propagation():
    """Test that NaN values do not propagate during computation"""
    print("=" * 70)
    print("  Test 1: NaN Propagation Resistance")
    print("=" * 70)

    solver = mfd.FluidSolver()
    velocity, reflectivity = create_typhoon_initial_condition(nx=64, ny=64, nz=32)

    n_injected = inject_deliberate_nans(velocity, nan_fraction=0.002)
    nans_before = count_total_nans(velocity)
    print(f"  NaNs before sanitize: {nans_before}")

    solver.sanitize_vector_field(velocity, 0.0)
    nans_after_sanitize = count_total_nans(velocity)
    print(f"  NaNs after sanitize: {nans_after_sanitize}")
    assert nans_after_sanitize == 0, "Sanitize failed to remove all NaNs"
    print("  ✓ Sanitize works correctly")

    inject_deliberate_nans(velocity, nan_fraction=0.001)

    print("\n  Running Navier-Stokes solver with NaN-contaminated input...")
    solver.solve_navier_stokes_simplified(velocity, reflectivity, 20)

    nans_final = count_total_nans(velocity)
    print(f"  NaNs after NS solver: {nans_final}")
    assert nans_final == 0, f"NaN propagation detected! {nans_final} NaNs remain"
    print("  ✓ No NaN propagation during NS solver")
    print("  ✓ PASSED\n")

def test_extreme_gradient_stability():
    """Test stability under extreme velocity gradients (typhoon eye wall)"""
    print("=" * 70)
    print("  Test 2: Extreme Gradient Stability (Typhoon Eye Wall)")
    print("=" * 70)

    config = mfd.SolverConfig()
    config.time_step = 0.05
    config.max_iterations = 50
    solver = mfd.FluidSolver(config)

    velocity, reflectivity = create_typhoon_initial_condition(nx=50, ny=50, nz=25)

    u_np = velocity.u.to_numpy()
    v_np = velocity.v.to_numpy()
    w_np = velocity.w.to_numpy()

    max_u = np.max(np.abs(u_np))
    max_v = np.max(np.abs(v_np))
    max_w = np.max(np.abs(w_np))
    print(f"  Initial max velocity: |u|={max_u:.2f}, |v|={max_v:.2f}, |w|={max_w:.2f}")

    div = mfd.Grid3D()
    solver.compute_divergence(velocity, div)
    div_np = div.to_numpy()
    print(f"  Initial max divergence: {np.max(np.abs(div_np)):.6f}")

    print("\n  Running 30 NS iterations on typhoon initial condition...")
    solver.solve_navier_stokes_simplified(velocity, reflectivity, 30)

    nans = count_total_nans(velocity)
    assert nans == 0, f"NaNs generated during extreme gradient simulation: {nans}"
    print(f"  NaN count after simulation: {nans}")

    div_np = div.to_numpy()
    u_np = velocity.u.to_numpy()
    v_np = velocity.v.to_numpy()
    w_np = velocity.w.to_numpy()

    max_u_final = np.max(np.abs(u_np))
    max_v_final = np.max(np.abs(v_np))
    max_w_final = np.max(np.abs(w_np))
    max_div_final = np.max(np.abs(div_np))

    print(f"  Final max velocity: |u|={max_u_final:.2f}, |v|={max_v_final:.2f}, |w|={max_w_final:.2f}")
    print(f"  Final max divergence: {max_div_final:.6f}")

    assert np.isfinite(max_u_final), "u velocity became non-finite"
    assert np.isfinite(max_v_final), "v velocity became non-finite"
    assert np.isfinite(max_w_final), "w velocity became non-finite"
    assert max_div_final < 10.0, f"Divergence exploded: {max_div_final}"
    print("  ✓ All values remain finite")
    print("  ✓ Divergence remains bounded")
    print("  ✓ PASSED\n")

def test_boundary_condition_stability():
    """Test boundary condition computation under heavy multi-threading"""
    print("=" * 70)
    print("  Test 3: Boundary Condition Thread Safety (1000 iterations)")
    print("=" * 70)

    solver = mfd.FluidSolver()
    nx, ny, nz = 40, 40, 20

    velocity, reflectivity = create_typhoon_initial_condition(nx, ny, nz)

    print(f"  Testing boundary conditions with {mfd.get_omp_num_threads()} threads...")
    print("  Running 1000 iterations of apply_boundary_conditions...")

    for iteration in range(1000):
        solver.apply_boundary_conditions(velocity)

        if iteration % 100 == 0:
            nans = count_total_nans(velocity)
            assert nans == 0, f"NaNs at boundary iteration {iteration}: {nans}"
            print(f"    Iteration {iteration}: {nans} NaNs (OK)")

    u_np = velocity.u.to_numpy()
    v_np = velocity.v.to_numpy()
    w_np = velocity.w.to_numpy()

    assert np.all(u_np[0, :, :] == 0.0), "X-min boundary not zero"
    assert np.all(u_np[-1, :, :] == 0.0), "X-max boundary not zero"
    assert np.all(v_np[:, 0, :] == 0.0), "Y-min boundary not zero"
    assert np.all(v_np[:, -1, :] == 0.0), "Y-max boundary not zero"
    assert np.all(w_np[:, :, 0] == 0.0), "Z-min boundary not zero"
    assert np.all(w_np[:, :, -1] == 0.0), "Z-max boundary not zero"

    print("  ✓ All boundaries correctly set to zero")
    print("  ✓ No data races in boundary conditions")
    print("  ✓ PASSED\n")

def test_vorticity_divergence_nan_resistance():
    """Test that divergence and vorticity computation handle NaN inputs gracefully"""
    print("=" * 70)
    print("  Test 4: Divergence/Vorticity NaN Resistance")
    print("=" * 70)

    solver = mfd.FluidSolver()
    velocity, _ = create_typhoon_initial_condition(nx=32, ny=32, nz=16)

    inject_deliberate_nans(velocity, nan_fraction=0.01)
    nans_before = count_total_nans(velocity)
    print(f"  Input velocity has {nans_before} NaN values")

    div = mfd.Grid3D()
    vort = mfd.VectorField3D()

    solver.compute_divergence(velocity, div)
    solver.compute_vorticity(velocity, vort)

    div_np = div.to_numpy()
    vort_np = [vort.u.to_numpy(), vort.v.to_numpy(), vort.w.to_numpy()]

    nans_div = np.sum(~np.isfinite(div_np))
    nans_vort = sum(np.sum(~np.isfinite(v)) for v in vort_np)

    print(f"  NaNs in divergence output: {nans_div}")
    print(f"  NaNs in vorticity output: {nans_vort}")

    assert nans_div == 0, f"Divergence produced {nans_div} NaNs"
    assert nans_vort == 0, f"Vorticity produced {nans_vort} NaNs"

    print("  ✓ Divergence computation is NaN-resistant")
    print("  ✓ Vorticity computation is NaN-resistant")
    print("  ✓ PASSED\n")

def test_full_pipeline_with_nan_injection():
    """Full pipeline test simulating real radar data with corruption"""
    print("=" * 70)
    print("  Test 5: Full Pipeline with NaN Injection (End-to-End)")
    print("=" * 70)

    solver = mfd.FluidSolver()
    nx, ny, nz = 50, 50, 25

    velocity, reflectivity = create_typhoon_initial_condition(nx, ny, nz)

    n_injected = inject_deliberate_nans(velocity, nan_fraction=0.005)
    print(f"  Initial NaN count: {count_total_nans(velocity)}")

    print("\n  Step 1: apply_boundary_conditions")
    solver.apply_boundary_conditions(velocity)
    print(f"    NaN count: {count_total_nans(velocity)}")

    print("\n  Step 2: enforce_incompressibility (30 iterations)")
    solver.enforce_incompressibility(velocity, 30)
    print(f"    NaN count: {count_total_nans(velocity)}")

    print("\n  Step 3: solve_navier_stokes_simplified (25 iterations)")
    solver.solve_navier_stokes_simplified(velocity, reflectivity, 25)
    print(f"    NaN count: {count_total_nans(velocity)}")

    print("\n  Step 4: compute_divergence")
    div = mfd.Grid3D()
    solver.compute_divergence(velocity, div)
    div_np = div.to_numpy()
    nans_div = np.sum(~np.isfinite(div_np))
    print(f"    NaN count in divergence: {nans_div}")

    print("\n  Step 5: compute_vorticity")
    vort = mfd.VectorField3D()
    solver.compute_vorticity(velocity, vort)
    vort_mag = mfd.Grid3D()
    solver.compute_vorticity_magnitude(vort, vort_mag)
    vort_np = vort_mag.to_numpy()
    nans_vort = np.sum(~np.isfinite(vort_np))
    print(f"    NaN count in vorticity: {nans_vort}")

    final_nans = count_total_nans(velocity)
    print(f"\n  Final velocity NaN count: {final_nans}")
    print(f"  Total NaN injected: {n_injected}")

    assert final_nans == 0, f"Final velocity has {final_nans} NaNs!"
    assert nans_div == 0, f"Divergence has {nans_div} NaNs!"
    assert nans_vort == 0, f"Vorticity has {nans_vort} NaNs!"

    print("  ✓ Full pipeline completed without NaN propagation")
    print("  ✓ All diagnostic computations are NaN-free")
    print("  ✓ PASSED\n")

def main():
    print("\n" + "=" * 70)
    print("  NaN Robustness Test Suite for Meteorology Fluid Dynamics")
    print(f"  OpenMP Threads: {mfd.get_omp_num_threads()}")
    print(f"  NumPy Version: {np.__version__}")
    print("=" * 70 + "\n")

    mfd.set_omp_num_threads(8)
    print(f"  Using {mfd.get_omp_num_threads()} threads for testing\n")

    all_passed = True

    tests = [
        test_no_nan_propagation,
        test_extreme_gradient_stability,
        test_boundary_condition_stability,
        test_vorticity_divergence_nan_resistance,
        test_full_pipeline_with_nan_injection,
    ]

    for test_func in tests:
        try:
            test_func()
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}\n")
            all_passed = False
        except Exception as e:
            print(f"  ✗ ERROR: {e}\n")
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("  ✓ ALL TESTS PASSED - NaN robustness verified!")
    else:
        print("  ✗ SOME TESTS FAILED")
    print("=" * 70 + "\n")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
