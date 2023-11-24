#include "flang/Optimizer/CodeGen/CodeGen.h"

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/LowLevelIntrinsics.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace fir {
#define GEN_PASS_DEF_OPENMPFIRCONVERSIONSTOLLVM
#include "flang/Optimizer/CodeGen/CGPasses.h.inc"
} // namespace fir

using namespace fir;

#define DEBUG_TYPE "flang-codegen-openmp"

// fir::LLVMTypeConverter for converting to LLVM IR dialect types.
#include "flang/Optimizer/CodeGen/TypeConverter.h"

namespace {
/// A pattern that converts the region arguments in a single-region OpenMP
/// operation to the LLVM dialect. The body of the region is not modified and is
/// expected to either be processed by the conversion infrastructure or already
/// contain ops compatible with LLVM dialect types.
template <typename OpType>
class OpenMPFIROpConversion : public mlir::ConvertOpToLLVMPattern<OpType> {
public:
  explicit OpenMPFIROpConversion(const fir::LLVMTypeConverter &lowering)
      : mlir::ConvertOpToLLVMPattern<OpType>(lowering) {}

  const fir::LLVMTypeConverter &lowerTy() const {
    return *static_cast<const fir::LLVMTypeConverter *>(
        this->getTypeConverter());
  }
};

// FIR Op specific conversion for MapInfoOp that overwrites the default OpenMP
// Dialect lowering, this allows FIR specific lowering of types, required for
// descriptors of allocatables currently.
struct MapInfoOpConversion
    : public OpenMPFIROpConversion<mlir::omp::MapInfoOp> {
  using OpenMPFIROpConversion::OpenMPFIROpConversion;

  void generateImplicitDescriptorMaps(
      mlir::Value varPtrPtr, mlir::ValueRange boundsOps, mlir::Type boxTy,
      mlir::Type llvmBoxTy, mlir::ConversionPatternRewriter &rewriter) const {}

  mlir::LogicalResult
  matchAndRewrite(mlir::omp::MapInfoOp curOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    const mlir::TypeConverter *converter = getTypeConverter();

    llvm::errs() << "executing match and rewrite for MapInfoOpConversion \n";
    llvm::SmallVector<mlir::Type> resTypes;
    if (failed(converter->convertTypes(curOp->getResultTypes(), resTypes)))
      return mlir::failure();

    curOp.dump();

    mlir::Value a = adaptor.getOperands()[0];
    a.dump();

    if (adaptor.getVarPtrPtr()) {
      adaptor.getVarPtrPtr().dump();
    }
    // auto loc = boxaddr.getLoc();
    // if (auto argty = boxaddr.getVal().getType().dyn_cast<fir::BaseBoxType>())
    // {
    //   TypePair boxTyPair = getBoxTypePair(argty);
    //   rewriter.replaceOp(boxaddr,
    //                      getBaseAddrFromBox(loc, boxTyPair, a, rewriter));
    // } else {
    // }
    // Copy attributes of the curOp except for the typeAttr which should
    // be converted
    llvm::SmallVector<mlir::NamedAttribute> newAttrs;
    for (mlir::NamedAttribute attr : curOp->getAttrs()) {
      if (auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(attr.getValue())) {
        mlir::Type newAttr;
        if (fir::isPointerType(typeAttr.getValue()) ||
            fir::isAllocatableType(typeAttr.getValue()) ||
            fir::isAssumedShape(typeAttr.getValue())) {
          newAttr = lowerTy().convertBoxTypeAsStruct(
              mlir::cast<fir::BaseBoxType>(typeAttr.getValue()));
          generateImplicitDescriptorMaps(
              adaptor.getVarPtrPtr(), adaptor.getBounds(), typeAttr.getValue(),
              newAttr, rewriter);
        } else {
          newAttr = converter->convertType(typeAttr.getValue());
        }
        newAttrs.emplace_back(attr.getName(), mlir::TypeAttr::get(newAttr));
      } else {
        newAttrs.push_back(attr);
      }
    }
    // TODO: Remove the bounds from the original map
    rewriter.replaceOpWithNewOp<mlir::omp::MapInfoOp>(
        curOp, resTypes, adaptor.getOperands(), newAttrs);
    return mlir::success();
  }
};
} // namespace

namespace {
class OpenMPFIRConversionsToLLVM
    : public fir::impl::OpenMPFIRConversionsToLLVMBase<
          OpenMPFIRConversionsToLLVM> {
public:
  OpenMPFIRConversionsToLLVM() {}

  inline mlir::ModuleOp getModule() { return getOperation(); }

  // possible to make overloads for the FIR types so they either remap all map
  // operands that are used or don't do a huge amount of alloca spawning?

  void runOnOperation() override final {
    fir::LLVMTypeConverter typeConverter{getModule(), /*applyTBAA*/ false,
                                         /*forceUnifiedTBAATree*/ false};
    mlir::IRRewriter rewriter(getModule()->getContext());
    getModule().dump();
    getModule().walk([&](mlir::Operation *op) {
      // FIR Op specific conversion for MapInfoOp's containing BoxTypes that are
      // descriptors this allows FIR specific lowering of types, required for
      // descriptors of allocatables currently.
      if (auto mapInfoOp = mlir::dyn_cast<mlir::omp::MapInfoOp>(op)) {
        if (mapInfoOp.getVarType().has_value() &&
            (fir::isPointerType(mapInfoOp.getVarType().value()) ||
             fir::isAllocatableType(mapInfoOp.getVarType().value()) ||
             fir::isAssumedShape(mapInfoOp.getVarType().value()))) {

          llvm::errs() << "printing in codegen \n";
          mapInfoOp.getVarPtr().dump();

          for (auto member : mapInfoOp.getMembers()) {
            member.dump();
          }

          llvm::errs() << "printing at end of codegen \n";
          mapInfoOp.setVarType(typeConverter.convertBoxTypeAsStruct(
              mlir::cast<fir::BaseBoxType>(mapInfoOp.getVarType().value())));
        }
      }
    });
  };
};
} // namespace

void fir::populateOpenMPFIRToLLVMConversionPatterns(
    LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns) {

  patterns.add<MapInfoOpConversion>(converter);
}

std::unique_ptr<mlir::Pass> fir::createOpenMPFIRConversionsToLLVMPass() {
  return std::make_unique<OpenMPFIRConversionsToLLVM>();
}
