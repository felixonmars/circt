//===- SeparateModules.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace circt;
namespace circt {
namespace arc {
#define GEN_PASS_DEF_ARCSEPARATEMODULES
#include "circt/Dialect/Arc/ArcPasses.h.inc"

struct ArcSeparateModules : public impl::ArcSeparateModulesBase<ArcSeparateModules> {
  void runOnOperation() override;
};

void ArcSeparateModules::runOnOperation() {
  auto module = getOperation();
  auto& sbt = getAnalysis<mlir::SymbolTable>();
  SmallVector<func::FuncOp> functions;
  for(auto& op : module.getOps()){
    auto func = dyn_cast<mlir::func::FuncOp>(op);
    if(!func)
      return signalPassFailure();
    functions.push_back(func);
  }

  for(auto func: functions){
    OpBuilder builder(func);
    auto mod = builder.create<mlir::ModuleOp>(func.getLoc());
    for(auto func2: functions){
      if(func == func2)
        continue;
    }
  }
  // Use callable? 
}

} // namespace circt
}