enable_testing()

add_executable(run_tests test_lexer.cpp)
target_link_libraries(run_tests PRIVATE cool_lib)
add_test(NAME LexerTest COMMAND run_tests)

add_executable(run_parser_tests test_parser.cpp)
target_link_libraries(run_parser_tests PRIVATE cool_lib)
add_test(NAME ParserTest COMMAND run_parser_tests)

add_executable(run_semantic_tests test_semantics.cpp)
target_link_libraries(run_semantic_tests PRIVATE cool_lib)
add_test(NAME SemanticTest COMMAND run_semantic_tests)

add_executable(run_codegen_tests test_codegen.cpp)
target_link_libraries(run_codegen_tests PRIVATE cool_lib)
add_test(NAME CodegenTest COMMAND run_codegen_tests)
