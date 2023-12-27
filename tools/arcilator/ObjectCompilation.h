namespace mlir {
  class ModuleOp;
}

namespace circt {
  void compileLLVMModule(mlir::ModuleOp module);
}