add_library(cool_runtime STATIC
    src/runtime.c
)
target_include_directories(cool_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
# Link against pthread
find_package(Threads REQUIRED)
target_link_libraries(cool_runtime PRIVATE Threads::Threads)