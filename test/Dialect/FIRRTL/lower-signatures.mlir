// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-signatures))' %s | FileCheck --check-prefixes=CHECK %s

firrtl.circuit "Prop" {
  // CHECK-LABEL @Prop(out %y: !firrtl.string)
  firrtl.module @Prop(out %y: !firrtl.string) attributes {convention = #firrtl<convention scalarized>} {
    %0 = firrtl.string "test"
    // CHECK: firrtl.propassign
    firrtl.propassign %y, %0 : !firrtl.string
  }

  firrtl.module private @emptyVec(in %vi : !firrtl.vector<uint<4>, 0>, out %vo : !firrtl.vector<uint<4>, 0>) attributes {convention = #firrtl<convention scalarized>} {
    firrtl.strictconnect %vo, %vi : !firrtl.vector<uint<4>, 0>
  }

  // CHECK-LABEL: @Annos
  // CHECK-SAME: in %x: !firrtl.uint<1> [{class = "circt.test", pin = "pin0"}],
  // CHECK-SAME: in %y_a: !firrtl.uint<1> [{class = "circt.test", pin = "pin1"}],
  // CHECK-SAME: in %y_b: !firrtl.uint<2> [{class = "circt.test", pin = "pin2"}])
  firrtl.module private @Annos(
    in %x: !firrtl.uint<1> [{circt.fieldID = 0 : i64, class = "circt.test", pin = "pin0"}],
    in %y: !firrtl.bundle<a: uint<1>, b: uint<2>> [{circt.fieldID = 2 : i64, class = "circt.test", pin = "pin2"}, {circt.fieldID = 1 : i64, class = "circt.test", pin = "pin1"}]
  )  attributes {convention = #firrtl<convention scalarized>} { 
  }

}
