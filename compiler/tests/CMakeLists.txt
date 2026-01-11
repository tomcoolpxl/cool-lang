enable_testing()

add_executable(run_tests test_lexer.cpp)
target_link_libraries(run_tests PRIVATE cool_lib)
add_test(NAME LexerTest COMMAND run_tests)

add_executable(run_parser_tests test_parser.cpp)
target_link_libraries(run_parser_tests PRIVATE cool_lib)
add_test(NAME ParserTest COMMAND run_parser_tests)