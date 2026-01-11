cmake_minimum_required(VERSION 3.20)
project(cool-lang LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Enable code coverage
add_compile_options(--coverage)
add_link_options(--coverage)

add_subdirectory(src)
add_subdirectory(tests)