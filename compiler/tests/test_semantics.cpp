#include "TestFramework.h"
#include "../src/lexer/Lexer.h"
#include "../src/parser/Parser.h"
#include "../src/semantics/SemanticAnalyzer.h"
#include <sstream>

// Helper to capture stderr since TestFramework doesn't support it
class CaptureStderr {
public:
    CaptureStderr() {
        old_buffer = std::cerr.rdbuf();
        std::cerr.rdbuf(buffer.rdbuf());
    }
    ~CaptureStderr() {
        std::cerr.rdbuf(old_buffer);
    }
    std::string getOutput() const {
        return buffer.str();
    }
private:
    std::stringstream buffer;
    std::streambuf* old_buffer;
};

TEST(test_semantic_analysis_basic) {
    std::string source = "fn main():\n    return 1\n";
    cool::Lexer lexer(source);
    cool::Parser parser(lexer);
    auto prog = parser.parseProgram();
    
    cool::SemanticAnalyzer analyzer;
    bool result = analyzer.analyze(*prog);
    ASSERT(result);
}

TEST(test_semantic_analysis_undefined_var) {
    std::string source = "fn main():\n    return x\n";
    cool::Lexer lexer(source);
    cool::Parser parser(lexer);
    auto prog = parser.parseProgram();
    
    cool::SemanticAnalyzer analyzer;
    CaptureStderr capture;
    bool result = analyzer.analyze(*prog);
    ASSERT(!result);
    // Optional: check output contains "Undefined variable: x"
}

TEST(test_semantic_analysis_move_error) {
    std::string source = 
        "fn foo(move x: i32):\n    return 0\n"
        "fn main():\n    let y = 10\n    foo(move y)\n    return y\n";
    cool::Lexer lexer(source);
    cool::Parser parser(lexer);
    auto prog = parser.parseProgram();
    
    cool::SemanticAnalyzer analyzer;
    CaptureStderr capture;
    bool result = analyzer.analyze(*prog);
    ASSERT(!result);
    ASSERT_EQ(capture.getOutput().find("Use of moved value: y") != std::string::npos, true);
}

TEST(test_semantic_analysis_double_move) {
    auto func = std::make_unique<cool::FunctionDecl>("main");
    
    // let y = 10
    auto init = std::make_unique<cool::LiteralExpr>("10");
    auto let = std::make_unique<cool::LetStmt>("y", std::move(init));
    func->body.push_back(std::move(let));
    
    // call(move y, move y)
    auto call = std::make_unique<cool::CallExpr>("some_func");
    
    auto var1 = std::make_unique<cool::VariableExpr>("y");
    auto arg1 = std::make_unique<cool::Argument>(cool::Argument::Mode::Move, std::move(var1));
    call->args.push_back(std::move(arg1));

    auto var2 = std::make_unique<cool::VariableExpr>("y");
    auto arg2 = std::make_unique<cool::Argument>(cool::Argument::Mode::Move, std::move(var2));
    call->args.push_back(std::move(arg2));
    
    func->body.push_back(std::make_unique<cool::ExprStmt>(std::move(call)));
    
    cool::Program prog;
    prog.decls.push_back(std::move(func));
    
    // Mock "some_func" existence
    auto otherFunc = std::make_unique<cool::FunctionDecl>("some_func");
    prog.decls.push_back(std::move(otherFunc));

    cool::SemanticAnalyzer analyzer;
    CaptureStderr capture;
    bool result = analyzer.analyze(prog);
    std::string output = capture.getOutput();
    
    ASSERT(!result);
    ASSERT_EQ(output.find("Double move detected: y") != std::string::npos, true);
}

TEST(test_semantic_analysis_no_escape) {
    auto func = std::make_unique<cool::FunctionDecl>("test_escape");
    
    // Parameter 'v' is a view
    // Param(name, typeName, isMove, isInOut)
    cool::Param param("v", "view[i32]", false, false);
    func->params.push_back(param);
    
    // return v
    auto var = std::make_unique<cool::VariableExpr>("v");
    auto ret = std::make_unique<cool::ReturnStmt>(std::move(var));
    func->body.push_back(std::move(ret));
    
    cool::Program prog;
    prog.decls.push_back(std::move(func));

    cool::SemanticAnalyzer analyzer;
    CaptureStderr capture;
    bool result = analyzer.analyze(prog);
    std::string output = capture.getOutput();
    
    ASSERT(!result);
    ASSERT_EQ(output.find("Escape Error: Cannot return a View") != std::string::npos, true);
}

TEST(test_semantic_analysis_branch_consistency) {
    auto func = std::make_unique<cool::FunctionDecl>("test_branch");
    
    // let x = 10
    auto init = std::make_unique<cool::LiteralExpr>("10");
    auto let = std::make_unique<cool::LetStmt>("x", std::move(init));
    func->body.push_back(std::move(let));
    
    // if (1) { move x } else { }
    auto cond = std::make_unique<cool::LiteralExpr>("1");
    std::vector<std::unique_ptr<cool::Stmt>> thenBlock;
    
    // call(move x)
    auto call = std::make_unique<cool::CallExpr>("consume");
    auto var = std::make_unique<cool::VariableExpr>("x");
    auto arg = std::make_unique<cool::Argument>(cool::Argument::Mode::Move, std::move(var));
    call->args.push_back(std::move(arg));
    thenBlock.push_back(std::make_unique<cool::ExprStmt>(std::move(call)));
    
    auto ifStmt = std::make_unique<cool::IfStmt>(std::move(cond), std::move(thenBlock));
    func->body.push_back(std::move(ifStmt));
    
    // After if: return x
    auto retVar = std::make_unique<cool::VariableExpr>("x");
    auto ret = std::make_unique<cool::ReturnStmt>(std::move(retVar));
    func->body.push_back(std::move(ret));
    
    cool::Program prog;
    prog.decls.push_back(std::move(func));
    
    // Mock "consume"
    auto consumeFunc = std::make_unique<cool::FunctionDecl>("consume");
    prog.decls.push_back(std::move(consumeFunc));

    cool::SemanticAnalyzer analyzer;
    CaptureStderr capture;
    bool result = analyzer.analyze(prog);
    std::string output = capture.getOutput();
    
    ASSERT(!result);
    ASSERT_EQ(output.find("Use of potentially moved value") != std::string::npos, true);
}

TEST_MAIN()
