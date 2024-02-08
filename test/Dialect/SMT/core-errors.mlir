// RUN: circt-opt %s --split-input-file --verify-diagnostics

func.func @solver_isolated_from_above(%arg0: !smt.bool) {
  // expected-note @below {{required by region isolation constraints}}
  smt.solver() : () -> () {
    // expected-error @below {{using value defined outside the region}}
    smt.assert %arg0
  }
  return
}

// -----

func.func @no_smt_value_enters_solver(%arg0: !smt.bool) {
  // expected-error @below {{operand #0 must be variadic of any non-smt type, but got '!smt.bool'}}
  smt.solver(%arg0) : (!smt.bool) -> () {
  ^bb0(%arg1: !smt.bool):
    smt.assert %arg1
  }
  return
}

// -----

func.func @no_smt_value_exits_solver() {
  // expected-error @below {{result #0 must be variadic of any non-smt type, but got '!smt.bool'}}
  %0 = smt.solver() : () -> !smt.bool {
    %a = smt.declare_const "a" : !smt.bool
    smt.yield %a : !smt.bool
  }
  return
}

// -----

func.func @block_args_and_inputs_match() {
  // expected-error @below {{block argument types must match the types of the 'inputs'}}
  smt.solver() : () -> () {
    ^bb0(%arg0: i32):
  }
  return
}

// -----

func.func @solver_yield_operands_and_results_match() {
  // expected-error @below {{types of yielded values must match return values}}
  smt.solver() : () -> () {
    %1 = hw.constant 0 : i32
    smt.yield %1 : i32
  }
  return
}

// -----

func.func @check_yield_operands_and_results_match() {
  // expected-error @below {{types of yielded values in 'unsat' region must match return values}}
  %0 = smt.check sat {
    %1 = hw.constant 0 : i32
    smt.yield %1 : i32
  } unknown {
    %1 = hw.constant 0 : i32
    smt.yield %1 : i32
  } unsat { } -> i32
  return
}

// -----

func.func @check_yield_operands_and_results_match() {
  // expected-error @below {{types of yielded values in 'unknown' region must match return values}}
  %0 = smt.check sat {
    %1 = hw.constant 0 : i32
    smt.yield %1 : i32
  } unknown {
  } unsat {
    %1 = hw.constant 0 : i32
    smt.yield %1 : i32
  } -> i32
  return
}

// -----

func.func @check_yield_operands_and_results_match() {
  // expected-error @below {{types of yielded values in 'sat' region must match return values}}
  %0 = smt.check sat {
  } unknown {
    %1 = hw.constant 0 : i32
    smt.yield %1 : i32
  } unsat {
    %1 = hw.constant 0 : i32
    smt.yield %1 : i32
  } -> i32
  return
}

// -----

func.func @check_no_block_arguments() {
  // expected-error @below {{region #0 should have no arguments}}
  smt.check sat {
  ^bb0(%arg0: i32):
  } unknown {
  } unsat {
  }
  return
}

// -----

func.func @check_no_block_arguments() {
  // expected-error @below {{region #1 should have no arguments}}
  smt.check sat {
  } unknown {
  ^bb0(%arg0: i32):
  } unsat {
  }
  return
}

// -----

func.func @check_no_block_arguments() {
  // expected-error @below {{region #2 should have no arguments}}
  smt.check sat {
  } unknown {
  } unsat {
  ^bb0(%arg0: i32):
  }
  return
}

// -----

func.func @too_few_operands() {
  // expected-error @below {{'inputs' must have at least size 2, but got 0}}
  smt.eq : !smt.bool
  return
}

// -----

func.func @too_few_operands(%a: !smt.bool) {
  // expected-error @below {{'inputs' must have at least size 2, but got 1}}
  smt.distinct %a : !smt.bool
  return
}

// -----

func.func @ite_type_mismatch(%a: !smt.bool, %b: !smt.bv<32>) {
  // expected-error @below {{failed to verify that all of {thenValue, elseValue, result} have same type}}
  "smt.ite"(%a, %a, %b) {} : (!smt.bool, !smt.bool, !smt.bv<32>) -> !smt.bool
  return
}

// -----

func.func @forall_number_of_decl_names_must_match_num_args() {
  // expected-error @below {{number of bound variable names must match number of block arguments}}
  %1 = smt.forall ["a"] {
  ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %2 = smt.eq %arg2, %arg3 : !smt.int
    smt.yield %2 : !smt.bool
  }
  return
}

// -----

func.func @exists_number_of_decl_names_must_match_num_args() {
  // expected-error @below {{number of bound variable names must match number of block arguments}}
  %1 = smt.exists ["a"] {
  ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %2 = smt.eq %arg2, %arg3 : !smt.int
    smt.yield %2 : !smt.bool
  }
  return
}

// -----

func.func @forall_yield_must_have_exactly_one_bool_value() {
  // expected-error @below {{yielded value must be of '!smt.bool' type}}
  %1 = smt.forall ["a", "b"] {
  ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %2 = smt.int.add %arg2, %arg3
    smt.yield %2 : !smt.int
  }
  return
}

// -----

func.func @forall_yield_must_have_exactly_one_bool_value() {
  // expected-error @below {{must have exactly one yielded value}}
  %1 = smt.forall ["a", "b"] {
  ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    smt.yield
  }
  return
}

// -----

func.func @exists_yield_must_have_exactly_one_bool_value() {
  // expected-error @below {{yielded value must be of '!smt.bool' type}}
  %1 = smt.exists ["a", "b"] {
  ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %2 = smt.int.add %arg2, %arg3
    smt.yield %2 : !smt.int
  }
  return
}

// -----

func.func @exists_yield_must_have_exactly_one_bool_value() {
  // expected-error @below {{must have exactly one yielded value}}
  %1 = smt.exists ["a", "b"] {
  ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    smt.yield
  }
  return
}

// -----

func.func @exists_patterns_region_and_no_patterns_attr_are_mutually_exclusive() {
  // expected-error @below {{patterns and the no_pattern attribute must not be specified at the same time}}
  %1 = smt.exists ["a"] no_pattern {
  ^bb0(%arg2: !smt.bool):
    smt.yield %arg2 : !smt.bool
  } patterns {
  ^bb0(%arg2: !smt.bool):
    smt.yield %arg2 : !smt.bool
  }
  return
}

// -----

func.func @forall_patterns_region_and_no_patterns_attr_are_mutually_exclusive() {
  // expected-error @below {{patterns and the no_pattern attribute must not be specified at the same time}}
  %1 = smt.forall ["a"] no_pattern {
  ^bb0(%arg2: !smt.bool):
    smt.yield %arg2 : !smt.bool
  } patterns {
  ^bb0(%arg2: !smt.bool):
    smt.yield %arg2 : !smt.bool
  }
  return
}

// -----

func.func @exists_patterns_region_num_args() {
  // expected-error @below {{block argument number and types of the 'body' and 'patterns' regions must match}}
  %1 = smt.exists ["a"] {
  ^bb0(%arg2: !smt.bool):
    smt.yield %arg2 : !smt.bool
  } patterns {
  ^bb0(%arg2: !smt.bool, %arg3: !smt.bool):
    smt.yield %arg2, %arg3 : !smt.bool, !smt.bool
  }
  return
}

// -----

func.func @forall_patterns_region_num_args() {
  // expected-error @below {{block argument number and types of the 'body' and 'patterns' regions must match}}
  %1 = smt.forall ["a"] {
  ^bb0(%arg2: !smt.bool):
    smt.yield %arg2 : !smt.bool
  } patterns {
  ^bb0(%arg2: !smt.bool, %arg3: !smt.bool):
    smt.yield %arg2, %arg3 : !smt.bool, !smt.bool
  }
  return
}

// -----

func.func @exists_patterns_region_at_least_one_yielded_value() {
  // expected-error @below {{'patterns' region must have at least one yielded value}}
  %1 = smt.exists ["a"] {
  ^bb0(%arg2: !smt.bool):
    smt.yield %arg2 : !smt.bool
  } patterns {
  ^bb0(%arg2: !smt.bool):
    smt.yield
  }
  return
}

// -----

func.func @forall_patterns_region_at_least_one_yielded_value() {
  // expected-error @below {{'patterns' region must have at least one yielded value}}
  %1 = smt.forall ["a"] {
  ^bb0(%arg2: !smt.bool):
    smt.yield %arg2 : !smt.bool
  } patterns {
  ^bb0(%arg2: !smt.bool):
    smt.yield
  }
  return
}

// -----

func.func @exists_patterns_region_no_non_smt_operations() {
  // expected-error @below {{'patterns' region may only contain SMT dialect operations}}
  %1 = smt.exists ["a"] {
  ^bb0(%arg2: !smt.bool):
    smt.yield %arg2 : !smt.bool
  } patterns {
  ^bb0(%arg2: !smt.bool):
    // expected-note @below {{first non-SMT operation here}}
    hw.constant 0 : i32
    smt.yield %arg2 : !smt.bool
  }
  return
}

// -----

func.func @forall_patterns_region_no_non_smt_operations() {
  // expected-error @below {{'patterns' region may only contain SMT dialect operations}}
  %1 = smt.forall ["a"] {
  ^bb0(%arg2: !smt.bool):
    smt.yield %arg2 : !smt.bool
  } patterns {
  ^bb0(%arg2: !smt.bool):
    // expected-note @below {{first non-SMT operation here}}
    hw.constant 0 : i32
    smt.yield %arg2 : !smt.bool
  }
  return
}

// -----

func.func @exists_patterns_region_no_var_binding_operations() {
  // expected-error @below {{'patterns' region must not contain any variable binding operations}}
  %1 = smt.exists ["a"] {
  ^bb0(%arg2: !smt.bool):
    smt.yield %arg2 : !smt.bool
  } patterns {
  ^bb0(%arg2: !smt.bool):
    // expected-note @below {{first violating operation here}}
    smt.exists ["b"] {
    ^bb0(%arg3: !smt.bool):
      smt.yield %arg3 : !smt.bool
    }
    smt.yield %arg2 : !smt.bool
  }
  return
}

// -----

func.func @forall_patterns_region_no_var_binding_operations() {
  // expected-error @below {{'patterns' region must not contain any variable binding operations}}
  %1 = smt.forall ["a"] {
  ^bb0(%arg2: !smt.bool):
    smt.yield %arg2 : !smt.bool
  } patterns {
  ^bb0(%arg2: !smt.bool):
    // expected-note @below {{first violating operation here}}
    smt.forall ["b"] {
    ^bb0(%arg3: !smt.bool):
      smt.yield %arg3 : !smt.bool
    }
    smt.yield %arg2 : !smt.bool
  }
  return
}

// -----

func.func @exists_patterns_region_block_args_used_at_least_once() {
  // expected-error @below {{every block argument in the 'patterns' region must be used at least once}}
  %1 = smt.exists ["a", "b"] {
  ^bb0(%arg2: !smt.bool, %arg3: !smt.bool):
    smt.yield %arg2 : !smt.bool
  } patterns {
  // expected-note @below {{first unused argument here}}
  ^bb0(%arg2: !smt.bool, %arg3: !smt.bool):
    smt.yield %arg2 : !smt.bool
  }
  return
}

// -----

func.func @forall_patterns_region_block_args_used_at_least_once() {
  // expected-error @below {{every block argument in the 'patterns' region must be used at least once}}
  %1 = smt.forall ["a", "b"] {
  ^bb0(%arg2: !smt.bool, %arg3: !smt.bool):
    smt.yield %arg2 : !smt.bool
  } patterns {
  // expected-note @below {{first unused argument here}}
  ^bb0(%arg2: !smt.bool, %arg3: !smt.bool):
    smt.yield %arg2 : !smt.bool
  }
  return
}
