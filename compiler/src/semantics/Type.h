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
    
    // Returns true if this type is a View (Transient) and cannot escape the stack.
    virtual bool isTransient() const { return false; }
    
    // Returns true if this type allows implicit copy (i.e. not linear).
    virtual bool isCopy() const { return false; }
};

struct PrimitiveType : Type {
    PrimitiveType(std::string n) : Type(TypeKind::Primitive, n) {}
    
    bool isCopy() const override {
        return name == "i32" || name == "i64" || name == "bool" || name == "void";
    }
};

struct StructType : Type {
    struct Field {
        std::string name;
        std::shared_ptr<Type> type;
    };
    std::vector<Field> fields;
    
    StructType(std::string n) : Type(TypeKind::Struct, n) {}

    bool isTransient() const override {
        for (const auto& field : fields) {
            if (field.type && field.type->isTransient()) {
                return true;
            }
        }
        return false;
    }
};

struct ViewType : Type {
    std::shared_ptr<Type> innerType;
    
    ViewType(std::shared_ptr<Type> inner) 
        : Type(TypeKind::Primitive, "view[" + inner->toString() + "]"), innerType(inner) {}
        
    bool isTransient() const override { return true; }
};

struct ChannelType : Type {
    std::shared_ptr<Type> innerType;
    
    ChannelType(std::shared_ptr<Type> inner) 
        : Type(TypeKind::Generic, "Channel[" + inner->toString() + "]"), innerType(inner) {}
        
    bool isCopy() const override { return false; }
};

class TypeRegistry {
public:
    static std::shared_ptr<Type> Int32() { static auto t = std::make_shared<PrimitiveType>("i32"); return t; }
    static std::shared_ptr<Type> Int64() { static auto t = std::make_shared<PrimitiveType>("i64"); return t; }
    static std::shared_ptr<Type> Bool() { static auto t = std::make_shared<PrimitiveType>("bool"); return t; }
    static std::shared_ptr<Type> String() { static auto t = std::make_shared<PrimitiveType>("str"); return t; }
    static std::shared_ptr<Type> Void() { static auto t = std::make_shared<PrimitiveType>("void"); return t; }
    
    static std::shared_ptr<Type> View(std::shared_ptr<Type> inner) {
        return std::make_shared<ViewType>(inner);
    }
};

} // namespace cool
