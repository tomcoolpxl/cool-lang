#pragma once
#include <string>
#include <memory>
#include <vector>

namespace cool {

enum class TypeKind {
    Primitive, // i32, bool, void
    Struct,    // User defined
    Protocol,  // Interfaces
    Generic    // List[T]
};

struct Type {
    TypeKind kind;
    std::string name;

    Type(TypeKind k, std::string n) : kind(k), name(n) {}
    virtual ~Type() = default;
    
    virtual bool equals(const Type& other) const {
        return name == other.name; 
    }
    
    virtual std::string toString() const { return name; }
};

struct PrimitiveType : Type {
    PrimitiveType(std::string n) : Type(TypeKind::Primitive, n) {}
};

struct StructType : Type {
    struct Field {
        std::string name;
        std::shared_ptr<Type> type;
    };
    std::vector<Field> fields;
    
    StructType(std::string n) : Type(TypeKind::Struct, n) {}
};

class TypeRegistry {
public:
    static std::shared_ptr<Type> Int32() { static auto t = std::make_shared<PrimitiveType>("i32"); return t; }
    static std::shared_ptr<Type> Int64() { static auto t = std::make_shared<PrimitiveType>("i64"); return t; }
    static std::shared_ptr<Type> Bool() { static auto t = std::make_shared<PrimitiveType>("bool"); return t; }
    static std::shared_ptr<Type> String() { static auto t = std::make_shared<PrimitiveType>("str"); return t; }
    static std::shared_ptr<Type> Void() { static auto t = std::make_shared<PrimitiveType>("void"); return t; }
};

} // namespace cool
