namespace mlir {
  class ModuleOp;
  class LogicalResult;
}

namespace circt {
  mlir::LogicalResult compileLLVMModule(mlir::ModuleOp module);
}