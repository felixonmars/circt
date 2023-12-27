#include "ObjectCompilation.h"
#include "mlir/IR/Threading.h"

#include "mlir/IR/BuiltinOps.h"
void circt::compileLLVMModule(mlir::ModuleOp module){
    auto* context = module.getContext();
    mlir::parallelForEach(&context);

}
