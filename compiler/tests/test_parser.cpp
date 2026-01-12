#include "TestFramework.h"
#include "../src/lexer/Lexer.h"
#include "../src/parser/Parser.h"

TEST(test_parse_simple_function) {
    std::string source = "fn main():\n    return 123\n";
    cool::Lexer lexer(source);
    cool::Parser parser(lexer);
    
    auto prog = parser.parseProgram();
    ASSERT(prog != nullptr);
    ASSERT(prog->decls.size() == 1);
}

TEST(test_parse_struct) {
    std::string source = "struct Point:\n    x: i32\n";
    cool::Lexer lexer(source);
    cool::Parser parser(lexer);
    auto prog = parser.parseProgram();
    ASSERT(prog->decls.size() == 1);
    ASSERT(dynamic_cast<cool::StructDecl*>(prog->decls[0].get()) != nullptr);
}

TEST(test_parse_let) {
    std::string source = "fn main():\n    let x = 1\n";
    cool::Lexer lexer(source);
    cool::Parser parser(lexer);
    auto prog = parser.parseProgram();
    ASSERT(prog->decls.size() == 1);
    auto func = dynamic_cast<cool::FunctionDecl*>(prog->decls[0].get());
    ASSERT(func != nullptr);
    ASSERT(func->body.size() == 1);
    ASSERT(dynamic_cast<cool::LetStmt*>(func->body[0].get()) != nullptr);
}

TEST(test_parse_call) {
    std::string source = "fn main():\n    foo(x, move y)\n";
    cool::Lexer lexer(source);
    cool::Parser parser(lexer);
    auto prog = parser.parseProgram();
    auto func = dynamic_cast<cool::FunctionDecl*>(prog->decls[0].get());
    auto stmt = dynamic_cast<cool::ExprStmt*>(func->body[0].get());
    auto call = dynamic_cast<cool::CallExpr*>(stmt->expr.get());
    ASSERT(call != nullptr);
    ASSERT(call->args.size() == 2);
    ASSERT(call->args[1]->mode == cool::Argument::Mode::Move);
}

TEST(test_parse_if_else) {
    std::string source = 
        "fn main():\n"
        "    if x:\n"
        "        return 1\n"
        "    else:\n"
        "        return 0\n";
    cool::Lexer lexer(source);
    cool::Parser parser(lexer);
    auto prog = parser.parseProgram();
    auto func = dynamic_cast<cool::FunctionDecl*>(prog->decls[0].get());
    ASSERT(dynamic_cast<cool::IfStmt*>(func->body[0].get()) != nullptr);
}

TEST(test_parse_while) {
    std::string source = 
        "fn main():\n"
        "    while 1:\n"
        "        print()\n";
    cool::Lexer lexer(source);
    cool::Parser parser(lexer);
    auto prog = parser.parseProgram();
    auto func = dynamic_cast<cool::FunctionDecl*>(prog->decls[0].get());
    ASSERT(dynamic_cast<cool::WhileStmt*>(func->body[0].get()) != nullptr);
}

TEST(test_parse_member_access) {
    std::string source = 
        "fn main():\n"
        "    x.y\n"
        "    x.y.z\n"
        "    x.method()\n";
    cool::Lexer lexer(source);
    cool::Parser parser(lexer);
    auto prog = parser.parseProgram();
    auto func = dynamic_cast<cool::FunctionDecl*>(prog->decls[0].get());
    
    // x.y
    auto stmt1 = dynamic_cast<cool::ExprStmt*>(func->body[0].get());
    auto mem1 = dynamic_cast<cool::MemberAccessExpr*>(stmt1->expr.get());
    ASSERT(mem1 != nullptr);
    ASSERT(mem1->member == "y");
    
    // x.y.z
    auto stmt2 = dynamic_cast<cool::ExprStmt*>(func->body[1].get());
    auto mem2 = dynamic_cast<cool::MemberAccessExpr*>(stmt2->expr.get());
    ASSERT(mem2 != nullptr);
    ASSERT(mem2->member == "z");
    auto inner = dynamic_cast<cool::MemberAccessExpr*>(mem2->object.get());
    ASSERT(inner->member == "y");
    
    // x.method()
    auto stmt3 = dynamic_cast<cool::ExprStmt*>(func->body[2].get());
    auto call = dynamic_cast<cool::CallExpr*>(stmt3->expr.get());
    ASSERT(call != nullptr);
    auto callee = dynamic_cast<cool::MemberAccessExpr*>(call->callee.get());
    ASSERT(callee != nullptr);
    ASSERT(callee->member == "method");
}

TEST_MAIN()