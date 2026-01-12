# Execution Verification Tests
# Real compilation and runtime verification of Cool programs

include_directories(../..)

add_executable(run_execution_verification_tests
    test_execution_real.cpp
)

target_link_libraries(run_execution_verification_tests
    cool_lib
)

add_test(NAME ExecutionVerificationTest COMMAND run_execution_verification_tests)

set_tests_properties(ExecutionVerificationTest PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
