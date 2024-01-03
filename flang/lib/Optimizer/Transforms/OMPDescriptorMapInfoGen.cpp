#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace fir {
#define GEN_PASS_DEF_OMPDESCRIPTORMAPINFOGENPASS
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

namespace {
class OMPDescriptorMapInfoGenPass
    : public fir::impl::OMPDescriptorMapInfoGenPassBase<
          OMPDescriptorMapInfoGenPass> {

  mlir::omp::MapInfoOp
  createMapInfoOp(fir::FirOpBuilder &builder, mlir::Location loc,
                  mlir::Value baseAddr, mlir::Value varPtrPtr, std::string name,
                  mlir::SmallVector<mlir::Value> bounds,
                  mlir::SmallVector<mlir::Value> members, uint64_t mapType,
                  mlir::omp::VariableCaptureKind mapCaptureType,
                  mlir::Type retTy, bool isVal = false) {
    if (auto boxTy = baseAddr.getType().dyn_cast<fir::BaseBoxType>()) {
      baseAddr = builder.create<fir::BoxAddrOp>(loc, baseAddr);
      retTy = baseAddr.getType();
    }

    mlir::TypeAttr varType = mlir::TypeAttr::get(
        llvm::cast<mlir::omp::PointerLikeType>(retTy).getElementType());

    mlir::omp::MapInfoOp op = builder.create<mlir::omp::MapInfoOp>(
        loc, retTy, baseAddr, varType, varPtrPtr, members, bounds,
        builder.getIntegerAttr(builder.getIntegerType(64, false), mapType),
        builder.getAttr<mlir::omp::VariableCaptureKindAttr>(mapCaptureType),
        builder.getStringAttr(name));

    return op;
  }

  void genDescriptorMemberMaps(mlir::omp::MapInfoOp op,
                               fir::FirOpBuilder &builder) {
    llvm::SmallVector<mlir::Value> descriptorBaseAddrMembers;
    mlir::Location loc = builder.getUnknownLoc();
    mlir::Value descriptor = op.getVarPtr();

    // If we enter this function, but the mapped type itself is not the
    // descriptor, then it's likely the address of the descriptor so we
    // must retrieve the descriptor SSA.
    if (!fir::isTypeWithDescriptor(op.getVarType())) {
      if (auto addrOp = mlir::dyn_cast_if_present<fir::BoxAddrOp>(
              op.getVarPtr().getDefiningOp())) {
        descriptor = addrOp.getVal();
      }
    }

    // The fir::BoxOffsetOp only works with !fir.ref<!fir.box<...>> types, as
    // allowing it to access non-reference box operations can cause some
    // problematic SSA IR. However, in the case of assumed shape's the type
    // is not a !fir.ref, in these cases to retrieve the appropriate
    // !fir.ref<!fir.box<...>> to access the data we need to map we must
    // perform an alloca and then store to it and retrieve the data from the new
    // alloca.
    if (mlir::isa<fir::BaseBoxType>(descriptor.getType())) {
      mlir::OpBuilder::InsertPoint insPt = builder.saveInsertionPoint();
      builder.setInsertionPointToStart(builder.getAllocaBlock());
      auto alloca = builder.create<fir::AllocaOp>(loc, descriptor.getType());
      builder.restoreInsertionPoint(insPt);
      builder.create<fir::StoreOp>(loc, descriptor, alloca);
      descriptor = alloca;
    }

    mlir::Value baseAddrAddr = builder.create<fir::BoxOffsetOp>(
        loc, descriptor, fir::BoxFieldAttr::base_addr);

    descriptorBaseAddrMembers.push_back(createMapInfoOp(
        builder, loc, baseAddrAddr, {}, "", op.getBounds(), {},
        op.getMapType().value(), mlir::omp::VariableCaptureKind::ByRef,
        fir::unwrapRefType(baseAddrAddr.getType())));

    // TODO: map the addendum segment of the descriptor, similarly to the above
    // base address/data pointer member.

    op.getVarPtrMutable().assign(descriptor);
    op.setVarType(fir::unwrapRefType(descriptor.getType()));
    op.getMembersMutable().assign(descriptorBaseAddrMembers);
    op.getBoundsMutable().assign(llvm::SmallVector<mlir::Value>{});
  }

  // This pass executes on mlir::ModuleOp's finding omp::MapInfoOp's containing
  // descriptor based types (allocatables, pointers, assumed shape etc.) and
  // expanding them into multiple omp::MapInfoOp's for each pointer member
  // contained within the descriptor.
  void runOnOperation() override {
    fir::KindMapping kindMap = fir::getKindMapping(getOperation());
    fir::FirOpBuilder builder{getOperation(), std::move(kindMap)};

    getOperation()->walk([&](mlir::omp::MapInfoOp op) {
      if (fir::isTypeWithDescriptor(op.getVarType()) ||
          mlir::isa<fir::BoxAddrOp>(op.getVarPtr().getDefiningOp())) {
        builder.setInsertionPoint(op);
        genDescriptorMemberMaps(op, builder);
      }
    });
  }
};

} // namespace

namespace fir {
std::unique_ptr<mlir::Pass> createOMPDescriptorMapInfoGenPass() {
  return std::make_unique<OMPDescriptorMapInfoGenPass>();
}
} // namespace fir
