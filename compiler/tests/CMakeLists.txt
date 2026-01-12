enable_testing()

# Unit Tests
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

add_executable(run_fn_keyword_tests test_fn_keyword.cpp)
target_link_libraries(run_fn_keyword_tests PRIVATE cool_lib)
add_test(NAME FnKeywordTest COMMAND run_fn_keyword_tests)

# Integration Test Categories
add_subdirectory(mlir_verification)
add_subdirectory(execution_verification)