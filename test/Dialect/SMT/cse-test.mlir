// RUN: circt-opt %s --cse | FileCheck %s

func.func @declare_const_cse(%in: i8) -> (!smt.bool, !smt.bool){
  // CHECK: smt.declare "a" : !smt.bool
  %a = smt.declare "a" : !smt.bool
  // CHECK-NEXT: smt.declare "a" : !smt.bool
  %b = smt.declare "a" : !smt.bool
  // CHECK-NEXT: return
  %c = smt.declare "a" : !smt.bool

  return %a, %b : !smt.bool, !smt.bool
}
