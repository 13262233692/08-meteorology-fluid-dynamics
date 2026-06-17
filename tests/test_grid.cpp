#include "grid3d.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace mfd;

int main() {
    std::cout << "=== Grid3D Tests ===" << std::endl;

    Grid3D g(64, 64, 32, 1.0, 1.0, 1.0);
    assert(g.nx() == 64);
    assert(g.ny() == 64);
    assert(g.nz() == 32);
    assert(g.size() == 64 * 64 * 32);
    std::cout << "[PASS] Grid allocation and dimensions" << std::endl;

    g.fill(42.0);
    for (size_t k = 0; k < g.nz(); ++k) {
        for (size_t j = 0; j < g.ny(); ++j) {
            for (size_t i = 0; i < g.nx(); ++i) {
                assert(std::abs(g(i, j, k) - 42.0) < 1e-12);
            }
        }
    }
    std::cout << "[PASS] Grid fill and access" << std::endl;

    g(10, 20, 15) = 99.0;
    assert(std::abs(g(10, 20, 15) - 99.0) < 1e-12);
    std::cout << "[PASS] Grid element write" << std::endl;

    size_t idx = g.index(10, 20, 15);
    assert(g.data()[idx] == 99.0);
    std::cout << "[PASS] Grid index calculation and raw data access" << std::endl;

    Grid3D g2 = g;
    assert(g2.nx() == g.nx() && g2.ny() == g.ny() && g2.nz() == g.nz());
    assert(std::abs(g2(10, 20, 15) - 99.0) < 1e-12);
    std::cout << "[PASS] Grid copy constructor" << std::endl;

    VectorField3D vf(32, 32, 16);
    vf.u.fill(1.0);
    vf.v.fill(2.0);
    vf.w.fill(3.0);
    assert(std::abs(vf.u(5, 5, 5) - 1.0) < 1e-12);
    assert(std::abs(vf.v(5, 5, 5) - 2.0) < 1e-12);
    assert(std::abs(vf.w(5, 5, 5) - 3.0) < 1e-12);
    std::cout << "[PASS] VectorField3D allocation and fill" << std::endl;

    assert(&vf.component(0) == &vf.u);
    assert(&vf.component(1) == &vf.v);
    assert(&vf.component(2) == &vf.w);
    std::cout << "[PASS] VectorField3D component access" << std::endl;

    g.clear();
    assert(g.size() == 0);
    assert(g.nx() == 0);
    std::cout << "[PASS] Grid clear" << std::endl;

    std::cout << "\n=== All Grid3D Tests PASSED ===" << std::endl;
    return 0;
}
