#include <iostream>
#include "lexer/Lexer.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: coolc <file.cool>" << std::endl;
        return 1;
    }

    std::cout << "Coolscript Compiler v0.1.0" << std::endl;
    // TODO: Connect Lexer
    return 0;
}
