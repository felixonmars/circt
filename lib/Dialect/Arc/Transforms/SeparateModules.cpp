//===- SeparateModules.cpp
//--------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace circt;
namespace circt {
namespace arc {
#define GEN_PASS_DEF_ARCSEPARATEMODULES
#include "circt/Dialect/Arc/ArcPasses.h.inc"

struct ArcSeparateModules
    : public impl::ArcSeparateModulesBase<ArcSeparateModules> {
  void runOnOperation() override;
};

void ArcSeparateModules::runOnOperation() {
  auto module = getOperation();
  module.dump();
  using FuncTy = mlir::LLVM::LLVMFuncOp;
  SmallVector<FuncTy> functions;
  for (auto &op : module.getOps()) {
    auto func = dyn_cast<FuncTy>(op);
    if (!func)
      return signalPassFailure();
    functions.push_back(func);
  }

  for (auto func : functions) {
    OpBuilder builder(func);
    if (func.empty())
      continue;
    auto mod = builder.create<mlir::ModuleOp>(func.getLoc());
    builder.setInsertionPoint(mod.getBody(), mod.getBody()->begin());
    auto cloned = func.clone();
    cloned.setPublic();
    builder.insert(cloned);

    for (auto func2 : llvm::reverse(functions)) {
      if (func == func2)
        continue;
      // TODO: Dont't clone unused functions.
      auto cloned = func2.clone();
      if (!cloned.empty()) {
        cloned.eraseBody();
        cloned.setPrivate();
      }
      builder.insert(cloned);
    }
  }

  for (auto func : functions) {
    func.erase();
  }
}

} // namespace arc
} // namespace circt