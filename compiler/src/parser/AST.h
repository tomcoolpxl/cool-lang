#pragma once
#include <string>
#include <vector>
#include <memory>
#include <iostream>

namespace cool {

// Forward declaration to avoid circular dependency
struct Type;

struct ASTNode {
    virtual ~ASTNode() = default;
    virtual void print(int indent = 0) const = 0;
};

// --- Expressions ---
struct Expr : ASTNode {
    std::shared_ptr<Type> resolvedType;
};

struct Argument : ASTNode {
    enum class Mode { View, Move, Copy, InOut };
    Mode mode;
    std::unique_ptr<Expr> expr;
    
    Argument(Mode m, std::unique_ptr<Expr> e) : mode(m), expr(std::move(e)) {}
    
    void print(int indent) const override {
        std::cout << std::string(indent, ' ') << "Arg (";
        switch(mode) {
            case Mode::View: std::cout << "view"; break;
            case Mode::Move: std::cout << "move"; break;
            case Mode::Copy: std::cout << "copy"; break;
            case Mode::InOut: std::cout << "inout"; break;
        }
        std::cout << "):\n";
        expr->print(indent + 2);
    }
};

struct LiteralExpr : Expr {
    std::string value;
    LiteralExpr(std::string v) : value(v) {}
    void print(int indent) const override {
        std::cout << std::string(indent, ' ') << "Literal: " << value << "\n";
    }
};

struct VariableExpr : Expr {
    std::string name;
    VariableExpr(std::string n) : name(n) {}
    void print(int indent) const override {
        std::cout << std::string(indent, ' ') << "Var: " << name << "\n";
    }
};

struct MemberAccessExpr : Expr {
    std::unique_ptr<Expr> object;
    std::string member;
    
    MemberAccessExpr(std::unique_ptr<Expr> o, std::string m) : object(std::move(o)), member(m) {}
    
    void print(int indent) const override {
        std::cout << std::string(indent, ' ') << "MemberAccess: ." << member << "\n";
        object->print(indent + 2);
    }
};

struct CallExpr : Expr {
    std::unique_ptr<Expr> callee;
    std::vector<std::unique_ptr<Argument>> args;
    
    CallExpr(std::unique_ptr<Expr> c) : callee(std::move(c)) {}
    
    void print(int indent) const override {
        std::cout << std::string(indent, ' ') << "Call:\n";
        callee->print(indent + 2);
        for (const auto& arg : args) {
            arg->print(indent + 2);
        }
    }
};

// --- Statements ---
struct Stmt : ASTNode {};

struct LetStmt : Stmt {
    std::string name;
    std::unique_ptr<Expr> initializer;
    
    LetStmt(std::string n, std::unique_ptr<Expr> i) : name(n), initializer(std::move(i)) {}
    
    void print(int indent) const override {
        std::cout << std::string(indent, ' ') << "Let: " << name << "\n";
        initializer->print(indent + 2);
    }
};

struct ReturnStmt : Stmt {
    std::unique_ptr<Expr> value;
    ReturnStmt(std::unique_ptr<Expr> v) : value(std::move(v)) {}
    void print(int indent) const override {
        std::cout << std::string(indent, ' ') << "Return\n";
        if (value) value->print(indent + 2);
    }
};

struct ExprStmt : Stmt {
    std::unique_ptr<Expr> expr;
    ExprStmt(std::unique_ptr<Expr> e) : expr(std::move(e)) {}
    void print(int indent) const override {
        std::cout << std::string(indent, ' ') << "ExprStmt\n";
        expr->print(indent + 2);
    }
};

struct IfStmt : Stmt {
    std::unique_ptr<Expr> condition;
    std::vector<std::unique_ptr<Stmt>> thenBlock;
    std::vector<std::unique_ptr<Stmt>> elseBlock;
    
    IfStmt(std::unique_ptr<Expr> cond, 
           std::vector<std::unique_ptr<Stmt>> thenB,
           std::vector<std::unique_ptr<Stmt>> elseB = {}) 
        : condition(std::move(cond)), thenBlock(std::move(thenB)), elseBlock(std::move(elseB)) {}
        
    void print(int indent) const override {
        std::cout << std::string(indent, ' ') << "If\n";
        condition->print(indent + 2);
        std::cout << std::string(indent + 1, ' ') << "Then:\n";
        for (const auto& stmt : thenBlock) stmt->print(indent + 2);
        if (!elseBlock.empty()) {
            std::cout << std::string(indent + 1, ' ') << "Else:\n";
            for (const auto& stmt : elseBlock) stmt->print(indent + 2);
        }
    }
};

struct WhileStmt : Stmt {
    std::unique_ptr<Expr> condition;
    std::vector<std::unique_ptr<Stmt>> body;
    
    WhileStmt(std::unique_ptr<Expr> cond, std::vector<std::unique_ptr<Stmt>> b) 
        : condition(std::move(cond)), body(std::move(b)) {}
        
    void print(int indent) const override {
        std::cout << std::string(indent, ' ') << "While\n";
        condition->print(indent + 2);
        std::cout << std::string(indent + 1, ' ') << "Body:\n";
        for (const auto& stmt : body) stmt->print(indent + 2);
    }
};

// --- Declarations ---
struct Param {
    std::string name;
    std::string typeName; // Type reference by name for now
    bool isMove;
    bool isInOut;
    
    Param(std::string n, std::string t, bool m, bool i) 
        : name(n), typeName(t), isMove(m), isInOut(i) {}
};

struct FunctionDecl : ASTNode {
    std::string name;
    std::vector<Param> params;
    std::string returnType;
    std::vector<std::unique_ptr<Stmt>> body;
    
    FunctionDecl(std::string n) : name(n) {}
    
    void print(int indent) const override {
        std::cout << std::string(indent, ' ') << "Function: " << name << "(";
        for (size_t i = 0; i < params.size(); ++i) {
            if (i > 0) std::cout << ", ";
            if (params[i].isMove) std::cout << "move ";
            if (params[i].isInOut) std::cout << "inout ";
            std::cout << params[i].name << ": " << params[i].typeName;
        }
        std::cout << ")" << (returnType.empty() ? "" : " -> " + returnType) << "\n";
        
        for (const auto& stmt : body) {
            stmt->print(indent + 2);
        }
    }
};

struct StructDecl : ASTNode {
    std::string name;
    std::vector<Param> fields; // Reuse Param for fields (name: type) logic
    
    StructDecl(std::string n) : name(n) {}
    
    void print(int indent) const override {
        std::cout << std::string(indent, ' ') << "Struct: " << name << "\n";
        for (const auto& field : fields) {
            std::cout << std::string(indent + 2, ' ') << field.name << ": " << field.typeName << "\n";
        }
    }
};

struct Program : ASTNode {
    std::vector<std::unique_ptr<ASTNode>> decls;
    
    void print(int indent = 0) const override {
        std::cout << "Program:\n";
        for (const auto& decl : decls) {
            decl->print(indent + 2);
        }
    }
};

} // namespace cool
