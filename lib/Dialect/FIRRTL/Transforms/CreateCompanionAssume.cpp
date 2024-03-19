//===- CreateCompanionAssume.cpp - Create an UNR only assume --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CreateCompanionAssume pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"

using namespace circt;
using namespace firrtl;

namespace {
struct CreateCompanionAssumePass
    : public CreateCompanionAssumeBase<CreateCompanionAssumePass> {
  void runOnOperation() override {
    getOperation().walk([](firrtl::AssertOp assertOp) {
      OpBuilder builder(assertOp);
      builder.setInsertionPointAfter(assertOp);
      auto guards = assertOp->getAttrOfType<ArrayAttr>("guards");
      auto assume = builder.create<firrtl::UNROnlyAssumeIntrinsicOp>(
          assertOp.getLoc(), assertOp.getClock(), assertOp.getPredicate(),
          assertOp.getEnable(), assertOp.getMessage(),
          assertOp.getSubstitutions(), assertOp.getName());
      if (guards)
        assume->setAttr("guards", guards);
    });
  }
};

} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::firrtl::createCreateCompanionAssume() {
  return std::make_unique<CreateCompanionAssumePass>();
}
