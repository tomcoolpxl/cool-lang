enable_testing()

add_executable(run_tests test_lexer.cpp)
target_link_libraries(run_tests PRIVATE cool_lib)

add_test(NAME LexerTest COMMAND run_tests)
