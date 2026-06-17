add_executable(test_solver test_solver.cpp)
target_link_libraries(test_solver PRIVATE mfd_core)
add_test(NAME test_solver COMMAND test_solver)

add_executable(test_grid test_grid.cpp)
target_link_libraries(test_grid PRIVATE mfd_core)
add_test(NAME test_grid COMMAND test_grid)
