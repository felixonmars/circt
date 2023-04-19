//===- DCToHW.cpp - Translate DC into HW ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This is the main DC to HW Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/DCToHW.h"
#include "../PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/DC/DCOps.h"
#include "circt/Dialect/DC/DCPasses.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/ValueMapper.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"
#include <optional>

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace circt;
using namespace circt::dc;
using namespace circt::hw;

using NameUniquer = std::function<std::string(Operation *)>;

// NOLINTNEXTLINE(misc-no-recursion)
static Type tupleToStruct(TupleType tuple) {
  auto *ctx = tuple.getContext();
  mlir::SmallVector<hw::StructType::FieldInfo, 8> hwfields;
  for (auto [i, innerType] : llvm::enumerate(tuple)) {
    Type convertedInnerType = innerType;
    if (auto tupleInnerType = innerType.dyn_cast<TupleType>())
      convertedInnerType = tupleToStruct(tupleInnerType);
    hwfields.push_back({StringAttr::get(ctx, "field" + std::to_string(i)),
                        convertedInnerType});
  }

  return hw::StructType::get(ctx, hwfields);
}

// Converts the range of 'types' into a `hw`-dialect type. The range will be
// converted to a `hw.struct` type.
static Type toHWType(Type t);
static Type toHWType(TypeRange types) {
  if (types.size() == 1)
    return toHWType(types.front());
  return toHWType(mlir::TupleType::get(types[0].getContext(), types));
}

// Converts any type 't' into a `hw`-compatible type.
// tuple -> hw.struct
// none -> i0
// (tuple[...] | hw.struct)[...] -> (tuple | hw.struct)[toHwType(...)]
static Type toHWType(Type t) {
  return TypeSwitch<Type, Type>(t)
      .Case<TupleType>(
          [&](TupleType tt) { return toHWType(tupleToStruct(tt)); })
      .Case<hw::StructType>([&](auto st) {
        llvm::SmallVector<hw::StructType::FieldInfo> structFields(
            st.getElements());
        for (auto &field : structFields)
          field.type = toHWType(field.type);
        return hw::StructType::get(st.getContext(), structFields);
      })
      .Case<NoneType>(
          [&](NoneType nt) { return IntegerType::get(nt.getContext(), 0); })
      .Default([&](Type t) { return t; });
}

static Type toESIHWType(Type t) {
  auto *ctx = t.getContext();
  Type outType =
      llvm::TypeSwitch<Type, Type>(t)
          .Case<ValueType>([&](auto vt) {
            return esi::ChannelType::get(ctx, toHWType(vt.getInnerTypes()));
          })
          .Case<TokenType>([&](auto tt) {
            return esi::ChannelType::get(ctx,
                                         IntegerType::get(tt.getContext(), 0));
          })
          .Default([](auto t) { return toHWType(t); });

  return outType;
}

namespace {

// Shared state used by various functions; captured in a struct to reduce the
// number of arguments that we have to pass around.
struct DCLoweringState {
  ModuleOp parentModule;
  NameUniquer nameUniquer;
};

// A type converter is needed to perform the in-flight materialization of "raw"
// (non-ESI channel) types to their ESI channel correspondents. This comes into
// effect when backedges exist in the input IR.
class ESITypeConverter : public TypeConverter {
public:
  ESITypeConverter() {
    addConversion([](Type type) -> Type { return toESIHWType(type); });
    addConversion([](esi::ChannelType t) -> Type { return t; });

    addTargetMaterialization(
        [&](mlir::OpBuilder &builder, mlir::Type resultType,
            mlir::ValueRange inputs,
            mlir::Location loc) -> std::optional<mlir::Value> {
          if (inputs.size() != 1)
            return std::nullopt;
          return inputs[0];
        });

    addSourceMaterialization(
        [&](mlir::OpBuilder &builder, mlir::Type resultType,
            mlir::ValueRange inputs,
            mlir::Location loc) -> std::optional<mlir::Value> {
          if (inputs.size() != 1)
            return std::nullopt;
          return inputs[0];
        });
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// HW Sub-module Related Functions
//===----------------------------------------------------------------------===//

namespace {

// Input handshakes contain a resolved valid and (optional )data signal, and
// a to-be-assigned ready signal.
struct InputHandshake {
  Value channel;
  Value valid;
  std::shared_ptr<Backedge> ready;
  Value data;
};

// Output handshakes contain a resolved ready, and to-be-assigned valid and
// (optional) data signals.
struct OutputHandshake {
  Value channel;
  std::shared_ptr<Backedge> valid;
  Value ready;
  std::shared_ptr<Backedge> data;
};

// Directly connect an input handshake to an output handshake
static void connect(InputHandshake &input, OutputHandshake &output) {
  output.valid->setValue(input.valid);
  input.ready->setValue(output.ready);
}

/// A helper struct that acts like a wire. Can be used to interact with the
/// RTLBuilder when multiple built components should be connected.
struct HandshakeWire {
  HandshakeWire(BackedgeBuilder &bb, Type dataType) {
    MLIRContext *ctx = dataType.getContext();
    auto i1Type = IntegerType::get(ctx, 1);
    valid = std::make_shared<Backedge>(bb.get(i1Type));
    ready = std::make_shared<Backedge>(bb.get(i1Type));
    data = std::make_shared<Backedge>(bb.get(dataType));
  }

  // Functions that allow to treat a wire like an input or output port.
  // **Careful**: Such a port will not be updated when backedges are resolved.
  InputHandshake getAsInput() {
    return InputHandshake{
        .channel = nullptr, .valid = *valid, .ready = ready, .data = *data};
  }
  OutputHandshake getAsOutput() {
    return OutputHandshake{
        .channel = nullptr, .valid = valid, .ready = *ready, .data = data};
  }

  std::shared_ptr<Backedge> valid;
  std::shared_ptr<Backedge> ready;
  std::shared_ptr<Backedge> data;
};

template <typename T, typename TInner>
llvm::SmallVector<T> extractValues(llvm::SmallVector<TInner> &container,
                                   llvm::function_ref<T(TInner &)> extractor) {
  llvm::SmallVector<T> result;
  llvm::transform(container, std::back_inserter(result), extractor);
  return result;
}
struct UnwrappedIO {
  llvm::SmallVector<InputHandshake> inputs;
  llvm::SmallVector<OutputHandshake> outputs;

  llvm::SmallVector<Value> getInputValids() {
    return extractValues<Value, InputHandshake>(
        inputs, [](auto &hs) { return hs.valid; });
  }
  llvm::SmallVector<std::shared_ptr<Backedge>> getInputReadys() {
    return extractValues<std::shared_ptr<Backedge>, InputHandshake>(
        inputs, [](auto &hs) { return hs.ready; });
  }
  llvm::SmallVector<std::shared_ptr<Backedge>> getOutputValids() {
    return extractValues<std::shared_ptr<Backedge>, OutputHandshake>(
        outputs, [](auto &hs) { return hs.valid; });
  }
  llvm::SmallVector<Value> getInputDatas() {
    return extractValues<Value, InputHandshake>(
        inputs, [](auto &hs) { return hs.data; });
  }
  llvm::SmallVector<Value> getOutputReadys() {
    return extractValues<Value, OutputHandshake>(
        outputs, [](auto &hs) { return hs.ready; });
  }

  llvm::SmallVector<Value> getOutputChannels() {
    return extractValues<Value, OutputHandshake>(
        outputs, [](auto &hs) { return hs.channel; });
  }
  llvm::SmallVector<std::shared_ptr<Backedge>> getOutputDatas() {
    return extractValues<std::shared_ptr<Backedge>, OutputHandshake>(
        outputs, [](auto &hs) { return hs.data; });
  }
};

// A class containing a bunch of syntactic sugar to reduce builder function
// verbosity.
// @todo: should be moved to support.
struct RTLBuilder {
  RTLBuilder(Location loc, OpBuilder &builder, Value clk = Value(),
             Value rst = Value())
      : b(builder), loc(loc), clk(clk), rst(rst) {}

  Value constant(const APInt &apv, std::optional<StringRef> name = {}) {
    // Cannot use zero-width APInt's in DenseMap's, see
    // https://github.com/llvm/llvm-project/issues/58013
    bool isZeroWidth = apv.getBitWidth() == 0;
    if (!isZeroWidth) {
      auto it = constants.find(apv);
      if (it != constants.end())
        return it->second;
    }

    auto cval = b.create<hw::ConstantOp>(loc, apv);
    if (!isZeroWidth)
      constants[apv] = cval;
    return cval;
  }

  Value constant(unsigned width, int64_t value,
                 std::optional<StringRef> name = {}) {
    return constant(APInt(width, value));
  }
  std::pair<Value, Value> wrap(Value data, Value valid,
                               std::optional<StringRef> name = {}) {
    auto wrapOp = b.create<esi::WrapValidReadyOp>(loc, data, valid);
    return {wrapOp.getResult(0), wrapOp.getResult(1)};
  }
  std::pair<Value, Value> unwrap(Value channel, Value ready,
                                 std::optional<StringRef> name = {}) {
    auto unwrapOp = b.create<esi::UnwrapValidReadyOp>(loc, channel, ready);
    return {unwrapOp.getResult(0), unwrapOp.getResult(1)};
  }

  // Various syntactic sugar functions.
  Value reg(StringRef name, Value in, Value rstValue, Value clk = Value(),
            Value rst = Value()) {
    Value resolvedClk = clk ? clk : this->clk;
    Value resolvedRst = rst ? rst : this->rst;
    assert(resolvedClk &&
           "No global clock provided to this RTLBuilder - a clock "
           "signal must be provided to the reg(...) function.");
    assert(resolvedRst &&
           "No global reset provided to this RTLBuilder - a reset "
           "signal must be provided to the reg(...) function.");

    return b.create<seq::CompRegOp>(loc, in.getType(), in, resolvedClk, name,
                                    resolvedRst, rstValue, StringAttr());
  }

  Value cmp(Value lhs, Value rhs, comb::ICmpPredicate predicate,
            std::optional<StringRef> name = {}) {
    return b.create<comb::ICmpOp>(loc, predicate, lhs, rhs);
  }

  Value buildNamedOp(llvm::function_ref<Value()> f,
                     std::optional<StringRef> name) {
    Value v = f();
    StringAttr nameAttr;
    Operation *op = v.getDefiningOp();
    if (name.has_value()) {
      op->setAttr("sv.namehint", b.getStringAttr(*name));
      nameAttr = b.getStringAttr(*name);
    }
    return v;
  }

  // Bitwise 'and'.
  Value bAnd(ValueRange values, std::optional<StringRef> name = {}) {
    return buildNamedOp(
        [&]() { return b.create<comb::AndOp>(loc, values, false); }, name);
  }

  Value bOr(ValueRange values, std::optional<StringRef> name = {}) {
    return buildNamedOp(
        [&]() { return b.create<comb::OrOp>(loc, values, false); }, name);
  }

  // Bitwise 'not'.
  Value bNot(Value value, std::optional<StringRef> name = {}) {
    auto allOnes = constant(value.getType().getIntOrFloatBitWidth(), -1);
    std::string inferedName;
    if (!name) {
      // Try to create a name from the input value.
      if (auto valueName =
              value.getDefiningOp()->getAttrOfType<StringAttr>("sv.namehint")) {
        inferedName = ("not_" + valueName.getValue()).str();
        name = inferedName;
      }
    }

    return buildNamedOp(
        [&]() { return b.create<comb::XorOp>(loc, value, allOnes); }, name);
  }

  Value shl(Value value, Value shift, std::optional<StringRef> name = {}) {
    return buildNamedOp(
        [&]() { return b.create<comb::ShlOp>(loc, value, shift); }, name);
  }

  Value concat(ValueRange values, std::optional<StringRef> name = {}) {
    return buildNamedOp([&]() { return b.create<comb::ConcatOp>(loc, values); },
                        name);
  }

  // Packs a list of values into a hw.struct.
  Value pack(ValueRange values, Type structType = Type(),
             std::optional<StringRef> name = {}) {
    if (!structType)
      structType = toHWType(values.getTypes());

    return buildNamedOp(
        [&]() { return b.create<hw::StructCreateOp>(loc, structType, values); },
        name);
  }

  // Unpacks a hw.struct into a list of values.
  ValueRange unpack(Value value) {
    auto structType = value.getType().cast<hw::StructType>();
    llvm::SmallVector<Type> innerTypes;
    structType.getInnerTypes(innerTypes);
    return b.create<hw::StructExplodeOp>(loc, innerTypes, value).getResults();
  }

  llvm::SmallVector<Value> toBits(Value v, std::optional<StringRef> name = {}) {
    llvm::SmallVector<Value> bits;
    for (unsigned i = 0, e = v.getType().getIntOrFloatBitWidth(); i != e; ++i)
      bits.push_back(b.create<comb::ExtractOp>(loc, v, i, /*bitWidth=*/1));
    return bits;
  }

  // OR-reduction of the bits in 'v'.
  Value rOr(Value v, std::optional<StringRef> name = {}) {
    return buildNamedOp([&]() { return bOr(toBits(v)); }, name);
  }

  // Extract bits v[hi:lo] (inclusive).
  Value extract(Value v, unsigned lo, unsigned hi,
                std::optional<StringRef> name = {}) {
    unsigned width = hi - lo + 1;
    return buildNamedOp(
        [&]() { return b.create<comb::ExtractOp>(loc, v, lo, width); }, name);
  }

  // Truncates 'value' to its lower 'width' bits.
  Value truncate(Value value, unsigned width,
                 std::optional<StringRef> name = {}) {
    return extract(value, 0, width - 1, name);
  }

  Value zext(Value value, unsigned outWidth,
             std::optional<StringRef> name = {}) {
    unsigned inWidth = value.getType().getIntOrFloatBitWidth();
    assert(inWidth <= outWidth && "zext: input width must be <- output width.");
    if (inWidth == outWidth)
      return value;
    auto c0 = constant(outWidth - inWidth, 0);
    return concat({c0, value}, name);
  }

  Value sext(Value value, unsigned outWidth,
             std::optional<StringRef> name = {}) {
    return comb::createOrFoldSExt(loc, value, b.getIntegerType(outWidth), b);
  }

  // Extracts a single bit v[bit].
  Value bit(Value v, unsigned index, std::optional<StringRef> name = {}) {
    return extract(v, index, index, name);
  }

  // Creates a hw.array of the given values.
  Value arrayCreate(ValueRange values, std::optional<StringRef> name = {}) {
    return buildNamedOp(
        [&]() { return b.create<hw::ArrayCreateOp>(loc, values); }, name);
  }

  // Extract the 'index'th value from the input array.
  Value arrayGet(Value array, Value index, std::optional<StringRef> name = {}) {
    return buildNamedOp(
        [&]() { return b.create<hw::ArrayGetOp>(loc, array, index); }, name);
  }

  // Muxes a range of values.
  // The select signal is expected to be a decimal value which selects starting
  // from the lowest index of value.
  Value mux(Value index, ValueRange values,
            std::optional<StringRef> name = {}) {
    if (values.size() == 2)
      return b.create<comb::MuxOp>(loc, index, values[1], values[0]);

    return arrayGet(arrayCreate(values), index, name);
  }

  // Muxes a range of values. The select signal is expected to be a 1-hot
  // encoded value.
  Value ohMux(Value index, ValueRange inputs) {
    // Confirm the select input can be a one-hot encoding for the inputs.
    unsigned numInputs = inputs.size();
    assert(numInputs == index.getType().getIntOrFloatBitWidth() &&
           "one-hot select can't mux inputs");

    // Start the mux tree with zero value.
    // Todo: clean up when handshake supports i0.
    auto dataType = inputs[0].getType();
    unsigned width =
        dataType.isa<NoneType>() ? 0 : dataType.getIntOrFloatBitWidth();
    Value muxValue = constant(width, 0);

    // Iteratively chain together muxes from the high bit to the low bit.
    for (size_t i = numInputs - 1; i != 0; --i) {
      Value input = inputs[i];
      Value selectBit = bit(index, i);
      muxValue = mux(selectBit, {muxValue, input});
    }

    return muxValue;
  }

  OpBuilder &b;
  Location loc;
  Value clk, rst;
  DenseMap<APInt, Value> constants;
};

/// Creates a Value that has an assigned zero value. For structs, this
/// corresponds to assigning zero to each element recursively.
static Value createZeroDataConst(RTLBuilder &s, Location loc, Type type) {
  return TypeSwitch<Type, Value>(type)
      .Case<NoneType>([&](NoneType) { return s.constant(0, 0); })
      .Case<IntType, IntegerType>([&](auto type) {
        return s.constant(type.getIntOrFloatBitWidth(), 0);
      })
      .Case<hw::StructType>([&](auto structType) {
        SmallVector<Value> zeroValues;
        for (auto field : structType.getElements())
          zeroValues.push_back(createZeroDataConst(s, loc, field.type));
        return s.b.create<hw::StructCreateOp>(loc, structType, zeroValues);
      })
      .Default([&](Type) -> Value {
        emitError(loc) << "unsupported type for zero value: " << type;
        assert(false);
        return {};
      });
}

static bool isZeroWidthType(Type type) {
  if (auto intType = type.dyn_cast<IntegerType>())
    return intType.getWidth() == 0;
  if (type.isa<NoneType>())
    return true;

  return false;
}

static UnwrappedIO unwrapIO(Location loc, ValueRange operands,
                            TypeRange results,
                            ConversionPatternRewriter &rewriter,
                            BackedgeBuilder &bb) {
  RTLBuilder rtlb(loc, rewriter);
  UnwrappedIO unwrapped;
  for (auto in : operands) {
    assert(isa<esi::ChannelType>(in.getType()));
    InputHandshake hs;
    auto ready = std::make_shared<Backedge>(bb.get(rtlb.b.getI1Type()));
    auto [data, valid] = rtlb.unwrap(in, *ready);
    hs.valid = valid;
    hs.ready = ready;
    hs.data = data;
    hs.channel = in;
    unwrapped.inputs.push_back(hs);
  }
  for (auto outputType : results) {
    outputType = toESIHWType(outputType);
    esi::ChannelType channelType = cast<esi::ChannelType>(outputType);
    OutputHandshake hs;
    Type innerType = channelType.getInner();
    Value data;
    if (isZeroWidthType(innerType)) {
      // Feed the ESI wrap with an i0 constant.
      data =
          rewriter.create<hw::ConstantOp>(loc, rewriter.getIntegerType(0), 0);
    } else {
      // Create a backedge for the unresolved data.
      auto dataBackedge = std::make_shared<Backedge>(bb.get(innerType));
      hs.data = dataBackedge;
      data = *dataBackedge;
    }
    auto valid = std::make_shared<Backedge>(bb.get(rewriter.getI1Type()));
    auto [dataCh, ready] = rtlb.wrap(data, *valid);
    hs.valid = valid;
    hs.ready = ready;
    hs.channel = dataCh;
    unwrapped.outputs.push_back(hs);
  }
  return unwrapped;
}

static UnwrappedIO unwrapIO(Operation *op, ValueRange operands,
                            ConversionPatternRewriter &rewriter,
                            BackedgeBuilder &bb) {
  return unwrapIO(op->getLoc(), operands, op->getResultTypes(), rewriter, bb);
}

// Returns the clock and reset values from the containing hw::HWModuleOp.
static std::pair<Value, Value> getClockAndReset(Operation *op) {
  hw::HWModuleOp parent = cast<hw::HWModuleOp>(op->getParentOp());
  size_t clockIdx = parent.getNumArguments() - 2;
  size_t resetIdx = parent.getNumArguments() - 1;

  return {parent.getArgument(clockIdx), parent.getArgument(resetIdx)};
}

class ForkConversionPattern : public OpConversionPattern<ForkOp> {
public:
  using OpConversionPattern<ForkOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ForkOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto bb = BackedgeBuilder(rewriter, op.getLoc());
    auto [clock, reset] = getClockAndReset(op);
    auto rtlb = RTLBuilder(op.getLoc(), rewriter, clock, reset);
    auto io = unwrapIO(op, operands.getOperands(), rewriter, bb);

    auto &input = io.inputs[0];

    auto c0I1 = rtlb.constant(1, 0);
    llvm::SmallVector<Value> doneWires;
    for (auto [i, output] : llvm::enumerate(io.outputs)) {
      auto doneBE = bb.get(rtlb.b.getI1Type());
      auto emitted = rtlb.bAnd({doneBE, rtlb.bNot(*input.ready)});
      auto emittedReg = rtlb.reg("emitted_" + std::to_string(i), emitted, c0I1);
      auto outValid = rtlb.bAnd({rtlb.bNot(emittedReg), input.valid});
      output.valid->setValue(outValid);
      auto validReady = rtlb.bAnd({output.ready, outValid});
      auto done =
          rtlb.bOr({validReady, emittedReg}, "done" + std::to_string(i));
      doneBE.setValue(done);
      doneWires.push_back(done);
    }
    input.ready->setValue(rtlb.bAnd(doneWires, "allDone"));

    rewriter.replaceOp(op, io.getOutputChannels());
    return success();
  }
};

class JoinConversionPattern : public OpConversionPattern<JoinOp> {
public:
  using OpConversionPattern<JoinOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(JoinOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto bb = BackedgeBuilder(rewriter, op.getLoc());
    auto io = unwrapIO(op, operands.getOperands(), rewriter, bb);
    auto rtlb = RTLBuilder(op.getLoc(), rewriter);
    auto &output = io.outputs[0];

    Value allValid = rtlb.bAnd(io.getInputValids());
    output.valid->setValue(allValid);

    auto validAndReady = rtlb.bAnd({output.ready, allValid});
    for (auto &input : io.inputs)
      input.ready->setValue(validAndReady);

    rewriter.replaceOp(op, io.outputs[0].channel);
    return success();
  }
};

class MergeConversionPattern : public OpConversionPattern<MergeOp> {
public:
  using OpConversionPattern<MergeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MergeOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto bb = BackedgeBuilder(rewriter, op.getLoc());
    auto io = unwrapIO(op, operands.getOperands(), rewriter, bb);
    auto rtlb = RTLBuilder(op.getLoc(), rewriter);

    // Extract select signal from the unwrapped IO.
    auto select = io.inputs[0];
    io.inputs.erase(io.inputs.begin());
    buildMuxLogic(rtlb, io, select);

    rewriter.replaceOp(op, io.outputs[0].channel);
    return success();
  }

  // Builds mux logic for the given inputs and outputs.
  // Note: it is assumed that the caller has removed the 'select' signal from
  // the 'unwrapped' inputs and provide it as a separate argument.
  void buildMuxLogic(RTLBuilder &s, UnwrappedIO &unwrapped,
                     InputHandshake &select) const {

    // ============================= Control logic =============================
    size_t numInputs = unwrapped.inputs.size();
    size_t selectWidth = llvm::Log2_64_Ceil(numInputs);
    Value truncatedSelect =
        select.data.getType().getIntOrFloatBitWidth() > selectWidth
            ? s.truncate(select.data, selectWidth)
            : select.data;

    // Decimal-to-1-hot decoder. 'shl' operands must be identical in size.
    auto selectZext = s.zext(truncatedSelect, numInputs);
    auto select1h = s.shl(s.constant(numInputs, 1), selectZext);
    auto &res = unwrapped.outputs[0];

    // Mux input valid signals.
    auto selectedInputValid =
        s.mux(truncatedSelect, unwrapped.getInputValids());
    // Result is valid when the selected input and the select input is valid.
    auto selAndInputValid = s.bAnd({selectedInputValid, select.valid});
    res.valid->setValue(selAndInputValid);
    auto resValidAndReady = s.bAnd({selAndInputValid, res.ready});

    // Select is ready when result is valid and ready (result transacting).
    select.ready->setValue(resValidAndReady);

    // Assign each input ready signal if it is currently selected.
    for (auto [inIdx, in] : llvm::enumerate(unwrapped.inputs)) {
      // Extract the selection bit for this input.
      auto isSelected = s.bit(select1h, inIdx);

      // '&' that with the result valid and ready, and assign to the input
      // ready signal.
      auto activeAndResultValidAndReady =
          s.bAnd({isSelected, resValidAndReady});
      in.ready->setValue(activeAndResultValidAndReady);
    }
  }
};

class ReturnConversionPattern : public OpConversionPattern<dc::ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(dc::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Locate existing output op, Append operands to output op, and move to
    // the end of the block.
    auto hwModule = cast<hw::HWModuleOp>(op->getParentOp());
    auto outputOp = *hwModule.getBodyBlock()->getOps<hw::OutputOp>().begin();
    outputOp->setOperands(adaptor.getOperands());
    outputOp->moveAfter(&hwModule.getBodyBlock()->back());
    rewriter.eraseOp(op);
    return success();
  }
};

class BranchConversionPattern : public OpConversionPattern<BranchOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(BranchOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto bb = BackedgeBuilder(rewriter, op.getLoc());
    auto io = unwrapIO(op, operands.getOperands(), rewriter, bb);
    auto rtlb = RTLBuilder(op.getLoc(), rewriter);
    auto cond = io.inputs[0];
    auto arg = io.inputs[1];
    auto trueRes = io.outputs[0];
    auto falseRes = io.outputs[1];

    auto condArgValid = rtlb.bAnd({cond.valid, arg.valid});

    // Connect valid signal of both results.
    trueRes.valid->setValue(rtlb.bAnd({cond.data, condArgValid}));
    falseRes.valid->setValue(rtlb.bAnd({rtlb.bNot(cond.data), condArgValid}));

    // Connect ready signal of input and condition.
    auto selectedResultReady =
        rtlb.mux(cond.data, {falseRes.ready, trueRes.ready});
    auto condArgReady = rtlb.bAnd({selectedResultReady, condArgValid});
    arg.ready->setValue(condArgReady);
    cond.ready->setValue(condArgReady);

    rewriter.replaceOp(
        op, llvm::SmallVector<Value>{trueRes.channel, falseRes.channel});
    return success();
  }
};

class SinkConversionPattern : public OpConversionPattern<SinkOp> {
public:
  using OpConversionPattern<SinkOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SinkOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto bb = BackedgeBuilder(rewriter, op.getLoc());
    auto io = unwrapIO(op, operands.getOperands(), rewriter, bb);
    io.inputs[0].ready->setValue(
        RTLBuilder(op.getLoc(), rewriter).constant(1, 1));
    rewriter.replaceOp(op, io.outputs[0].channel);
    return success();
  }
};

class SourceConversionPattern : public OpConversionPattern<SourceOp> {
public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto bb = BackedgeBuilder(rewriter, op.getLoc());
    auto io = unwrapIO(op, operands.getOperands(), rewriter, bb);
    auto rtlb = RTLBuilder(op.getLoc(), rewriter);
    io.outputs[0].valid->setValue(rtlb.constant(1, 1));
    io.outputs[0].data->setValue(rtlb.constant(0, 0));
    rewriter.replaceOp(op, io.outputs[0].channel);
    return success();
  }
};

class PackConversionPattern : public OpConversionPattern<PackOp> {
public:
  using OpConversionPattern<PackOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(PackOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto bb = BackedgeBuilder(rewriter, op.getLoc());
    auto io = unwrapIO(op, llvm::SmallVector<Value>{operands.getToken()},
                       rewriter, bb);
    auto rtlb = RTLBuilder(op.getLoc(), rewriter);
    auto &input = io.inputs[0];
    auto &output = io.outputs[0];

    Value packedData;
    if (operands.getInputs().size() > 1)
      packedData = rtlb.pack(operands.getInputs());
    else
      packedData = operands.getInputs()[0];

    output.data->setValue(packedData);
    connect(input, output);
    rewriter.replaceOp(op, output.channel);
    return success();
  }
};

class UnpackConversionPattern : public OpConversionPattern<UnpackOp> {
public:
  using OpConversionPattern<UnpackOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(UnpackOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto bb = BackedgeBuilder(rewriter, op.getLoc());
    auto io = unwrapIO(
        op.getLoc(), llvm::SmallVector<Value>{operands.getInput()},
        // Only generate an output channel for the token typed output.
        llvm::SmallVector<Type>{op.getToken().getType()}, rewriter, bb);
    auto rtlb = RTLBuilder(op.getLoc(), rewriter);
    auto &input = io.inputs[0];
    auto &output = io.outputs[0];

    llvm::SmallVector<Value> unpackedValues;
    if (op.getInput().getType().cast<ValueType>().getInnerTypes().size() != 1)
      unpackedValues = rtlb.unpack(input.data);
    else
      unpackedValues.push_back(input.data);

    connect(input, output);
    llvm::SmallVector<Value> outputs;
    outputs.push_back(output.channel);
    outputs.append(unpackedValues.begin(), unpackedValues.end());
    rewriter.replaceOp(op, outputs);
    return success();
  }
};

class BufferConversionPattern : public OpConversionPattern<BufferOp> {
public:
  using OpConversionPattern<BufferOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BufferOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto [clock, reset] = getClockAndReset(op);

    // ... esi.buffer should in theory provide a correct (latency-insensitive)
    // implementation...
    Type channelType = operands.getInput().getType();
    rewriter.replaceOpWithNewOp<esi::ChannelBufferOp>(
        op, channelType, clock, reset, operands.getInput(), op.getSizeAttr(),
        nullptr);
    return success();

    /*
      InputHandshake lastStage;
      SmallVector<int64_t> initValues;

      // For now, always build seq buffers.
      if (op.getInitValues())
        initValues = op.getInitValueArray();

      Type dataType;
      if (auto innerTypes = op.getInnerTypes(); innerTypes.has_value())
        dataType = toHWType(innerTypes.value());
      else {
        // The "data" is i0.4
        dataType = IntegerType::get(op.getContext(), 0);
      }

      lastStage = buildSeqBufferLogic(rtlb, bb, dataType, op.getSize(), input,
                                      output, initValues);

      // Connect the last stage to the output handshake.
      output.data->setValue(lastStage.data);
      output.valid->setValue(lastStage.valid);
      lastStage.ready->setValue(output.ready);

      rewriter.replaceOp(op, output.channel);
      return success();
      */
  };

  struct SeqBufferStage {
    SeqBufferStage(Type dataType, InputHandshake &preStage, BackedgeBuilder &bb,
                   RTLBuilder &s, size_t index,
                   std::optional<int64_t> initValue)
        : dataType(dataType), preStage(preStage), s(s), bb(bb), index(index) {
      // TODO: refactor all of this - this is just taken from HandshakeToFIRRTL
      // which is know to work (e.g. data and control registers are somewhat
      // intermingled). When there is no data register, we just create that as
      // an i0 registerring, which will get canonicalized away...

      currentStage.ready = std::make_shared<Backedge>(bb.get(s.b.getI1Type()));
      auto hasInitValue = s.constant(1, initValue.has_value());
      auto validBE = bb.get(s.b.getI1Type());
      auto validReg = s.reg(getRegName("valid"), validBE, hasInitValue);
      auto readyBE = bb.get(s.b.getI1Type());

      Value initValueCs;
      if (initValue.has_value())
        initValueCs = s.constant(dataType.getIntOrFloatBitWidth(), *initValue);
      else
        initValueCs = createZeroDataConst(s, s.loc, dataType);

      Value dataReg =
          buildDataBufferLogic(validReg, initValueCs, validBE, readyBE);

      buildControlBufferLogic(validReg, readyBE, dataReg);
    }

    StringAttr getRegName(StringRef name) {
      return s.b.getStringAttr(name + std::to_string(index) + "_reg");
    }

    void buildControlBufferLogic(Value validReg, Backedge &readyBE,
                                 Value dataReg) {
      auto c0I1 = s.constant(1, 0);
      auto readyRegWire = bb.get(s.b.getI1Type());
      auto readyReg = s.reg(getRegName("ready"), readyRegWire, c0I1);

      // Create the logic to drive the current stage valid and potentially
      // data.
      currentStage.valid = s.mux(readyReg, {validReg, readyReg},
                                 "controlValid" + std::to_string(index));

      // Create the logic to drive the current stage ready.
      auto notReadyReg = s.bNot(readyReg);
      readyBE.setValue(notReadyReg);

      auto succNotReady = s.bNot(*currentStage.ready);
      auto neitherReady = s.bAnd({succNotReady, notReadyReg});
      auto ctrlNotReady = s.mux(neitherReady, {readyReg, validReg});
      auto bothReady = s.bAnd({*currentStage.ready, readyReg});

      // Create a mux for emptying the register when both are ready.
      auto resetSignal = s.mux(bothReady, {ctrlNotReady, c0I1});
      readyRegWire.setValue(resetSignal);

      // Add same logic for the data path if necessary.
      auto ctrlDataRegBE = bb.get(dataType);
      auto ctrlDataReg = s.reg(getRegName("ctrl_data"), ctrlDataRegBE, c0s);
      auto dataResult = s.mux(readyReg, {dataReg, ctrlDataReg});
      currentStage.data = dataResult;

      auto dataNotReadyMux = s.mux(neitherReady, {ctrlDataReg, dataReg});
      auto dataResetSignal = s.mux(bothReady, {dataNotReadyMux, c0s});
      ctrlDataRegBE.setValue(dataResetSignal);
    }

    Value buildDataBufferLogic(Value validReg, Value initValue,
                               Backedge &validBE, Backedge &readyBE) {
      // Create a signal for when the valid register is empty or the successor
      // is ready to accept new token.
      auto notValidReg = s.bNot(validReg);
      auto emptyOrReady = s.bOr({notValidReg, readyBE});
      preStage.ready->setValue(emptyOrReady);

      // Create a mux that drives the register input. If the emptyOrReady
      // signal is asserted, the mux selects the predValid signal. Otherwise,
      // it selects the register output, keeping the output registered
      // unchanged.
      auto validRegMux = s.mux(emptyOrReady, {validReg, preStage.valid});

      // Now we can drive the valid register.
      validBE.setValue(validRegMux);

      // Create a mux that drives the date register.
      auto dataRegBE = bb.get(dataType);
      auto dataReg =
          s.reg(getRegName("data"),
                s.mux(emptyOrReady, {dataRegBE, preStage.data}), initValue);
      dataRegBE.setValue(dataReg);
      return dataReg;
    }

    InputHandshake getOutput() { return currentStage; }

    Type dataType;
    InputHandshake &preStage;
    InputHandshake currentStage;
    RTLBuilder &s;
    BackedgeBuilder &bb;
    size_t index;

    // A zero-valued constant of equal type as the data type of this buffer.
    Value c0s;
  };

  InputHandshake buildSeqBufferLogic(RTLBuilder &s, BackedgeBuilder &bb,
                                     Type dataType, unsigned size,
                                     InputHandshake &input,
                                     OutputHandshake &output,
                                     llvm::ArrayRef<int64_t> initValues) const {
    // Prime the buffer building logic with an initial stage, which just
    // wraps the input handshake.
    InputHandshake currentStage = input;

    for (unsigned i = 0; i < size; ++i) {
      bool isInitialized = i < initValues.size();
      auto initValue =
          isInitialized ? std::optional<int64_t>(initValues[i]) : std::nullopt;
      currentStage = SeqBufferStage(dataType, currentStage, bb, s, i, initValue)
                         .getOutput();
    }

    return currentStage;
  };
};

static hw::ModulePortInfo getModulePortInfo(dc::FuncOp funcOp) {
  hw::ModulePortInfo ports({}, {});
  auto *ctx = funcOp->getContext();
  auto ft = funcOp.getFunctionType();

  // Add all inputs of funcOp.
  unsigned inIdx = 0;
  for (auto [index, type] : llvm::enumerate(ft.getInputs())) {
    ports.inputs.push_back({StringAttr::get(ctx, "in" + std::to_string(index)),
                            hw::PortDirection::INPUT, toESIHWType(type), index,
                            hw::InnerSymAttr{}});
    inIdx++;
  }

  // Add all outputs of funcOp.
  for (auto [index, type] : llvm::enumerate(ft.getResults())) {
    ports.outputs.push_back({StringAttr::get(ctx, "in" + std::to_string(index)),
                             hw::PortDirection::OUTPUT, toESIHWType(type),
                             index, hw::InnerSymAttr{}});
  }

  // Add clock and reset signals.
  Type i1Type = IntegerType::get(ctx, 1);
  ports.inputs.push_back({StringAttr::get(ctx, "clock"),
                          hw::PortDirection::INPUT, i1Type, inIdx++,
                          hw::InnerSymAttr{}});
  ports.inputs.push_back({StringAttr::get(ctx, "reset"),
                          hw::PortDirection::INPUT, i1Type, inIdx,
                          hw::InnerSymAttr{}});

  return ports;
}

class FuncOpConversionPattern : public OpConversionPattern<dc::FuncOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(dc::FuncOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    ModulePortInfo ports = getModulePortInfo(op);

    if (op.isExternal()) {
      rewriter.create<hw::HWModuleExternOp>(
          op.getLoc(), rewriter.getStringAttr(op.getName()), ports);
    } else {
      auto hwModule = rewriter.create<hw::HWModuleOp>(
          op.getLoc(), rewriter.getStringAttr(op.getName()), ports);

      auto &region = op->getRegions().front();

      Region &moduleRegion = hwModule->getRegions().front();
      rewriter.mergeBlocks(
          &region.getBlocks().front(), hwModule.getBodyBlock(),
          hwModule.getBodyBlock()->getArguments().drop_back(2));
      TypeConverter::SignatureConversion result(moduleRegion.getNumArguments());
      (void)getTypeConverter()->convertSignatureArgs(
          moduleRegion.getArgumentTypes(), result);
      rewriter.applySignatureConversion(&moduleRegion, result);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// HW Top-module Related Functions
//===----------------------------------------------------------------------===//

static LogicalResult convertFuncOp(ESITypeConverter &typeConverter,
                                   ConversionTarget &target, dc::FuncOp op,
                                   OpBuilder &moduleBuilder) {

  std::map<std::string, unsigned> instanceNameCntr;

  RewritePatternSet patterns(op.getContext());

  patterns.insert<
      FuncOpConversionPattern, ReturnConversionPattern, ForkConversionPattern,
      JoinConversionPattern, MergeConversionPattern, BranchConversionPattern,
      PackConversionPattern, UnpackConversionPattern, BufferConversionPattern,
      SourceConversionPattern, SinkConversionPattern>(typeConverter,
                                                      op.getContext());

  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    return op->emitOpError() << "error during conversion";
  return success();
}

static LogicalResult allDCValuesHasOneUse(mlir::FunctionOpInterface funcOp) {
  if (funcOp.isExternal())
    return success();

  auto checkUseFunc = [&](Operation *op, Value v, Twine desc,
                          unsigned idx) -> LogicalResult {
    if (!v.getType().isa<TokenType, ValueType>())
      return success();

    auto numUses = std::distance(v.getUses().begin(), v.getUses().end());
    if (numUses == 0)
      return op->emitOpError() << desc << " " << idx << " has no uses.";
    if (numUses > 1)
      return op->emitOpError() << desc << " " << idx << " has multiple uses.";
    return success();
  };

  // Validate blocks within the function
  for (auto [idx, block] : enumerate(funcOp.getBlocks())) {
    // Validate ops within the block
    for (auto &subOp : block) {
      for (auto res : llvm::enumerate(subOp.getResults())) {
        if (failed(checkUseFunc(&subOp, res.value(), "result", res.index())))
          return failure();
      }
    }

    for (auto &barg : block.getArguments()) {
      if (failed(checkUseFunc(funcOp.getOperation(), barg,
                              "block #" + Twine(idx) + " argument",
                              barg.getArgNumber())))
        return failure();
    }
  }
  return success();
}

namespace {
class DCToHWPass : public DCToHWBase<DCToHWPass> {
public:
  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();

    // Lowering to HW requires that every DC-typed value is used exactly once.
    // Check whether this precondition is met, and if not, exit.
    for (auto f : mod.getOps<dc::FuncOp>()) {
      if (auto res = allDCValuesHasOneUse(f); failed(res)) {
        f.emitOpError() << "DCToHW: failed to verify that all values "
                           "are used exactly once. Remember to run the "
                           "fork/sink materialization pass before HW lowering.";
        signalPassFailure();
        return;
      }
    }

    ESITypeConverter typeConverter;
    ConversionTarget target(getContext());
    // All top-level logic of a handshake module will be the interconnectivity
    // between instantiated modules.
    target.addIllegalDialect<dc::DCDialect>();
    target.addLegalDialect<hw::HWDialect, comb::CombDialect, seq::SeqDialect,
                           esi::ESIDialect>();

    OpBuilder submoduleBuilder(mod.getContext());
    submoduleBuilder.setInsertionPointToStart(mod.getBody());
    for (auto f : llvm::make_early_inc_range(mod.getOps<dc::FuncOp>())) {
      if (failed(convertFuncOp(typeConverter, target, f, submoduleBuilder))) {
        signalPassFailure();
        return;
      }
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> circt::createDCToHWPass() {
  return std::make_unique<DCToHWPass>();
}