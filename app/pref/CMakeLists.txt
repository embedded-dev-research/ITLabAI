add_executable(timer_test timertest.cpp)

target_compile_options(timer_test PRIVATE "/openmp")
