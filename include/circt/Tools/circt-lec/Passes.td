//===-- Passes.td - LEC Transforms pass definition file ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for LEC transformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_CIRCT_LEC_PASSES_TD
#define CIRCT_TOOLS_CIRCT_LEC_PASSES_TD

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def ConstructLEC : Pass<"construct-lec", "::mlir::ModuleOp"> {
  let summary = "Lower CIRCTs core dialects to a LEC problem statement";
  let description = [{
    Lowers HW and Comb operations to the SMT and Func dialect. HW modules become
    functions that take SMT bit-vectors as input and return SMT bit-vectors
    according to the original ports of the module. The Comb operations in the
    module body are converted to equivalent SMT expressions. HW instance
    operations are not supported, run Arc's ModuleInliner pass beforehand
    (ideally the two modules to check equivalence are marked public and all the
    other ones are marked private).

    This pass also inserts a main function in LLVM dialect to initialize the SMT
    context and boilerplate SMT constraints to compare the two modules against
    each other.
  }];

  let options = [
    Option<"firstModule", "first-module", "std::string",
           /*default=*/"",
           "Name of the first of the two modules to compare.">,
    Option<"secondModule", "second-module", "std::string",
           /*default=*/"",
           "Name of the second of the two modules to compare.">,
    Option<"insertMainFunc", "insert-main", "bool",
           /*default=*/"false",
           "Whether a main function should be inserted for AOT compilation.">,
  ];

  let dependentDialects = [
    "verif::VerifDialect", "mlir::func::FuncDialect", "mlir::LLVM::LLVMDialect"
  ];
}

#endif // CIRCT_TOOLS_CIRCT_LEC_PASSES_TD

