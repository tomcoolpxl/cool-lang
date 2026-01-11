add_library(cool_lib OBJECT
    lexer/Lexer.cpp
    lexer/Token.cpp
)
target_include_directories(cool_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(coolc main.cpp)
target_link_libraries(coolc PRIVATE cool_lib)