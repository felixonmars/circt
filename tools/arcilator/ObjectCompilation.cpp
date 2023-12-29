#include "ObjectCompilation.h"
#include "mlir/IR/Threading.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/IR/BuiltinOps.h"
void circt::compileLLVMModule(mlir::ModuleOp module){
    auto* context = module.getContext();
    for(auto module: module.getOps<mlir::func::FuncOp>())
        {
            // TODO: Compile it.
        }
}
