#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include "Type.h"

namespace cool {

enum class OwnershipState {
    Owned,          // Alive and owned
    Burned,         // Moved out
    Borrowed,       // Read-only view active
    MutBorrowed     // Mutable view active (inout)
};

struct Symbol {
    std::string name;
    std::shared_ptr<Type> type;
    OwnershipState state;
};

class SymbolTable {
public:
    SymbolTable() { enterScope(); } 

    void enterScope() {
        scopes.push_back({});
    }

    void exitScope() {
        if (!scopes.empty()) scopes.pop_back();
    }

    bool define(std::string name, std::shared_ptr<Type> type) {
        auto& currentScope = scopes.back();
        if (currentScope.find(name) != currentScope.end()) {
            return false; 
        }
        currentScope[name] = Symbol{name, type, OwnershipState::Owned};
        return true;
    }

    Symbol* resolve(std::string name) {
        for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
            auto found = it->find(name);
            if (found != it->end()) {
                return &found->second;
            }
        }
        return nullptr;
    }

private:
    std::vector<std::unordered_map<std::string, Symbol>> scopes;
};

} // namespace cool
