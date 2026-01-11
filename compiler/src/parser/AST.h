#pragma once
#include <string>
#include <vector>
#include <memory>
#include <iostream>

namespace cool {

struct ASTNode {
    virtual ~ASTNode() = default;
    virtual void print(int indent = 0) const = 0;
};

// --- Expressions ---
struct Expr : ASTNode {};

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

// --- Statements ---
struct Stmt : ASTNode {};

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

// --- Declarations ---
struct FunctionDecl : ASTNode {
    std::string name;
    // TODO: Params
    std::vector<std::unique_ptr<Stmt>> body;
    
    FunctionDecl(std::string n) : name(n) {}
    
    void print(int indent) const override {
        std::cout << std::string(indent, ' ') << "Function: " << name << "\n";
        for (const auto& stmt : body) {
            stmt->print(indent + 2);
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
