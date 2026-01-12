# MLIR Pattern Verification Tests
# These tests verify MLIR intermediate representation generation
# by pattern matching on the generated IR string.
# They do NOT execute the code, only verify compilation to MLIR.

include_directories(../..)

add_executable(run_mlir_verification_tests
    test_runner.cpp
)

target_link_libraries(run_mlir_verification_tests
    cool_lib
)

add_test(NAME MLIRVerificationTest COMMAND run_mlir_verification_tests)

set_tests_properties(MLIRVerificationTest PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
