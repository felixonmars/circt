//===- PortConverter.h - Module I/O rewriting utility -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The PortConverter is a utility class for rewriting arguments of a
// HWMutableModuleLike operation.
// It is intended to be a generic utility that can facilitate replacement of
// a given module in- or output to an arbitrary set of new inputs and outputs
// (i.e. 1 port -> N in, M out ports). Typical usecases is where an in (or
// output) of a module represents some higher-level abstraction that will be
// implemented by a set of lower-level in- and outputs ports + supporting
// operations within a module. It also attempts to do so in an optimal way, by
// e.g. being able to collect multiple port modifications of a module, and
// perform them all at once.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_PORTCONVERTER_H
#define CIRCT_SUPPORT_PORTCONVERTER_H

#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {

class SignalStandardBuilder;
class SignalingStandard;

class PortConverterImpl {
public:
  /// Run port conversion.
  LogicalResult run();
  Block *getBody() const { return body; }
  hw::HWMutableModuleLike getModule() const { return mod; }

  /// These two methods take care of allocating new ports in the correct place
  /// based on the position of 'origPort'. The new port is based on the original
  /// name and suffix. The specification for the new port is given by `newPort`
  /// and is recorded internally. Any changes to 'newPort' after calling this
  /// will not be reflected in the modules new port list. Will also add the new
  /// input to the block arguments of the body of the module.
  Value createNewInput(hw::PortInfo origPort, const Twine &suffix, Type type,
                       hw::PortInfo &newPort);
  /// Same as above. 'output' is the value fed into the new port and is required
  /// if 'body' is non-null. Important note: cannot be a backedge which gets
  /// replaced since this isn't attached to an op until later in the pass.
  void createNewOutput(hw::PortInfo origPort, const Twine &suffix, Type type,
                       Value output, hw::PortInfo &newPort);

protected:
  PortConverterImpl(hw::InstanceGraphNode *moduleNode)
      : moduleNode(moduleNode) {
    mod = dyn_cast<hw::HWMutableModuleLike>(*moduleNode->getModule());
    assert(mod && "PortConverter only works on HWMutableModuleLike");

    if (mod->getNumRegions() == 1 && mod->getRegion(0).hasOneBlock())
      body = &mod->getRegion(0).front();
  }

  std::unique_ptr<SignalStandardBuilder> ssb;

private:
  /// Materializes/commits all of the recorded port changes to the module.
  void materializeChanges();

  /// Updates an instance of the module. This is called after the module has
  /// been updated. It will update the instance to match the new port
  void updateInstance(hw::InstanceOp);

  // If the module has a block and it wants to be modified, this'll be
  // nomoduleNoden-null.
  Block *body = nullptr;

  hw::InstanceGraphNode *moduleNode;
  hw::HWMutableModuleLike mod;

  // Keep around a reference to the specific signaling standard classes to
  // facilitate updating the instance ops. Indexed by the original port
  // location.
  SmallVector<std::unique_ptr<SignalingStandard>> loweredInputs;
  SmallVector<std::unique_ptr<SignalingStandard>> loweredOutputs;

  // Tracking information to modify the module. Populated by the
  // 'createNew(Input|Output)' methods. Will be cleared once port changes have
  // materialized. Default length is  0 to save memory in case we'll be keeping
  // this around for later use.
  SmallVector<std::pair<unsigned, hw::PortInfo>, 0> newInputs;
  SmallVector<std::pair<unsigned, hw::PortInfo>, 0> newOutputs;
  SmallVector<Value, 0> newOutputValues;
};

/// Base class for the signaling standard of a particular port. Abstracts the
/// details of a particular signaling standard from the port layout. Subclasses
/// keep around port mapping information to use when updating instances.
class SignalingStandard {
public:
  SignalingStandard(PortConverterImpl &converter, hw::PortInfo origPort)
      : converter(converter), body(converter.getBody()), origPort(origPort) {}
  virtual ~SignalingStandard() = default;

  // Lower the specified port into a wire-level signaling protocol. The two
  // virtual methods 'build*Signals' should be overridden by subclasses. They
  // should use the 'create*' methods in 'PortConverter' to create the
  // necessary ports.
  void lowerPort() {
    if (origPort.direction == hw::PortDirection::OUTPUT)
      buildOutputSignals();
    else
      buildInputSignals();
  }

  /// Update an instance port to the new port information.
  virtual void mapInputSignals(OpBuilder &b, Operation *inst, Value instValue,
                               SmallVectorImpl<Value> &newOperands,
                               ArrayRef<Backedge> newResults) = 0;
  virtual void mapOutputSignals(OpBuilder &b, Operation *inst, Value instValue,
                                SmallVectorImpl<Value> &newOperands,
                                ArrayRef<Backedge> newResults) = 0;

  MLIRContext *getContext() { return getModule()->getContext(); }
  bool isUntouched() const { return isUntouchedFlag; }

protected:
  // Build the input and output signals for the port. This pertains to modifying
  // the module itself.
  virtual void buildInputSignals() = 0;
  virtual void buildOutputSignals() = 0;

  PortConverterImpl &converter;
  Block *body;
  hw::PortInfo origPort;

  hw::HWMutableModuleLike getModule() { return converter.getModule(); }

  // We don't need full RTTI support for SignalingStandard, we only need to know
  // if this SignalingStandard is the UntouchedSignalingStandard.
  bool isUntouchedFlag = false;
};

/// We consider non-caught ports to be ad-hoc signaling or 'untouched'. (Which
/// counts as a signaling protocol if one squints pretty hard). We mostly do
/// this since it allows us a more consistent internal API.
class UntouchedSignalingStandard : public SignalingStandard {
public:
  UntouchedSignalingStandard(PortConverterImpl &converter,
                             hw::PortInfo origPort)
      : SignalingStandard(converter, origPort) {
    // Set the 'RTTI flag' to true.
    isUntouchedFlag = true;
  }

  void mapInputSignals(OpBuilder &b, Operation *inst, Value instValue,
                       SmallVectorImpl<Value> &newOperands,
                       ArrayRef<Backedge> newResults) override {
    newOperands[origPort.argNum] = instValue;
  }
  void mapOutputSignals(OpBuilder &b, Operation *inst, Value instValue,
                        SmallVectorImpl<Value> &newOperands,
                        ArrayRef<Backedge> newResults) override {
    instValue.replaceAllUsesWith(newOperands[origPort.argNum]);
  }

private:
  void buildInputSignals() override {
    Value newValue =
        converter.createNewInput(origPort, "", origPort.type, portInfo);
    if (body)
      body->getArgument(origPort.argNum).replaceAllUsesWith(newValue);
  }

  void buildOutputSignals() override {
    Value output;
    if (body)
      output = body->getTerminator()->getOperand(origPort.argNum);
    converter.createNewOutput(origPort, "", origPort.type, output, portInfo);
  }

  hw::PortInfo portInfo;
};

// A SignalStandardBuilder will, given an input type, build the appropriate
// signaling standard for that type.
class SignalStandardBuilder {
public:
  SignalStandardBuilder(PortConverterImpl &converter) : converter(converter) {}
  virtual ~SignalStandardBuilder() = default;

  // Builds the appropriate signaling standard for the port. Users should
  // override this method with their own llvm::TypeSwitch-based dispatch code,
  // and by default call this method when no signaling standard applies.
  virtual FailureOr<std::unique_ptr<SignalingStandard>>
  build(hw::PortInfo port) {
    // Default builder is the 'untouched' signaling standard..
    return {std::make_unique<UntouchedSignalingStandard>(converter, port)};
  }

  PortConverterImpl &converter;
};

// The PortConverter wraps a single HWMutableModuleLike operation, and is
// initialized from an instance graph node. The port converter is templated
// on a SignalStandardBuilder, which is used to build the appropriate
// signaling standard for each port type.
template <typename SignalStandardBuilderImpl>
class PortConverter : public PortConverterImpl {
public:
  PortConverter(hw::InstanceGraph &graph, hw::HWMutableModuleLike mod)
      : PortConverterImpl(graph.lookup(cast<hw::HWModuleLike>(*mod))) {
    ssb = std::make_unique<SignalStandardBuilderImpl>(*this);
  }
};

} // namespace circt

#endif // CIRCT_SUPPORT_PORTCONVERTER_H
