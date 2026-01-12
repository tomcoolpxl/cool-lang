#include <iostream>
#include <fstream>
#include <sstream>
#include <optional>

#include "lexer/Lexer.h"
#include "parser/Parser.h"
#include "semantics/SemanticAnalyzer.h"
#include "codegen/MLIRGenerator.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"

#include "llvm/Support/TargetSelect.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

std::string readFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << path << std::endl;
        exit(1);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: coolc <file.cool>" << std::endl;
        return 1;
    }

    std::string source = readFile(argv[1]);

    // 1. Lex & Parse
    cool::Lexer lexer(source);
    cool::Parser parser(lexer);
    
    std::cout << "-- Parsing --" << std::endl;
    auto program = parser.parseProgram();
    if (!program) {
        std::cerr << "Parsing failed." << std::endl;
        return 1;
    }

    // 2. Semantic Analysis
    std::cout << "-- Analyzing --" << std::endl;
    cool::SemanticAnalyzer analyzer;
    if (!analyzer.analyze(*program)) {
        std::cerr << "Semantic analysis failed." << std::endl;
        return 1;
    }

    // 3. Code Generation (MLIR)
    std::cout << "-- Generating MLIR --" << std::endl;
    cool::MLIRGenerator generator;
    std::string mlirCode = generator.generate(*program);
    
    std::cout << "\n=== Generated MLIR ===\n" << std::endl;
    std::cout << mlirCode << std::endl;

    // 4. Parse MLIR
    std::cout << "-- Parsing MLIR with LLVM --" << std::endl;
    mlir::MLIRContext context;
    context.allowUnregisteredDialects(true); 
    
    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect, mlir::cf::ControlFlowDialect, mlir::LLVM::LLVMDialect>();
    mlir::registerBuiltinDialectTranslation(registry);
    mlir::registerLLVMDialectTranslation(registry);
    context.appendDialectRegistry(registry);
    
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceString<mlir::ModuleOp>(mlirCode, &context);
    if (!module) {
        std::cerr << "Failed to parse MLIR" << std::endl;
        return 1;
    }
    
    if (failed(module->verify())) {
        std::cerr << "MLIR verification failed" << std::endl;
        module->dump();
        return 1;
    }
    
    std::cout << "MLIR Parsed Successfully!" << std::endl;

    // 5. Lower to LLVM Dialect
    std::cout << "-- Lowering to LLVM Dialect --" << std::endl;
    mlir::PassManager pm(&context);
    pm.addPass(mlir::createSCFToControlFlowPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    
    if (failed(pm.run(*module))) {
        std::cerr << "Lowering to LLVM Dialect failed" << std::endl;
        module->dump();
        return 1;
    }

    // 6. Translate to LLVM IR
    std::cout << "-- Translating to LLVM IR --" << std::endl;
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);
    if (!llvmModule) {
        std::cerr << "Translation to LLVM IR failed" << std::endl;
        return 1;
    }
    
    // 7. Emit Object File
    std::cout << "-- Emitting Object File --" << std::endl;
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();
    
    auto targetTriple = llvm::sys::getDefaultTargetTriple();
    llvmModule->setTargetTriple(llvm::Triple(targetTriple));
    
    std::string error;
    auto target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
    
    if (!target) {
        std::cerr << "Failed to lookup target: " << error << std::endl;
        return 1;
    }
    
    llvm::TargetOptions opt;
    auto rm = std::optional<llvm::Reloc::Model>();
    auto targetMachine = target->createTargetMachine(targetTriple, "generic", "", opt, rm);
    
    llvmModule->setDataLayout(targetMachine->createDataLayout());
    
    std::error_code ec;
    llvm::raw_fd_ostream dest("output.o", ec, llvm::sys::fs::OF_None);
    if (ec) {
        std::cerr << "Could not open file: " << ec.message() << std::endl;
        return 1;
    }
    
    llvm::legacy::PassManager pass;
    if (targetMachine->addPassesToEmitFile(pass, dest, nullptr, llvm::CodeGenFileType::ObjectFile)) {
        std::cerr << "TargetMachine can't emit a file of this type" << std::endl;
        return 1;
    }
    
    pass.run(*llvmModule);
    dest.flush();
    
    std::cout << "Success! Output written to output.o" << std::endl;

    return 0;
}
