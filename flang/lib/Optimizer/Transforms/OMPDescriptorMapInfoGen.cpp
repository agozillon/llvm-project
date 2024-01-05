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
#include <iterator>

namespace fir {
#define GEN_PASS_DEF_OMPDESCRIPTORMAPINFOGENPASS
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

namespace {
class OMPDescriptorMapInfoGenPass
    : public fir::impl::OMPDescriptorMapInfoGenPassBase<
          OMPDescriptorMapInfoGenPass> {

  void genDescriptorMemberMaps(mlir::omp::MapInfoOp op,
                               fir::FirOpBuilder &builder,
                               mlir::Operation *target) {
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

    descriptorBaseAddrMembers.push_back(builder.create<mlir::omp::MapInfoOp>(
        loc, baseAddrAddr.getType(), baseAddrAddr,
        llvm::cast<mlir::omp::PointerLikeType>(
            fir::unwrapRefType(baseAddrAddr.getType()))
            .getElementType(),
        mlir::Value{}, mlir::SmallVector<mlir::Value>{}, op.getBounds(),
        builder.getIntegerAttr(builder.getIntegerType(64, false),
                               op.getMapType().value()),
        builder.getAttr<mlir::omp::VariableCaptureKindAttr>(
            mlir::omp::VariableCaptureKind::ByRef),
        builder.getStringAttr("")));

    // TODO: map the addendum segment of the descriptor, similarly to the
    // above base address/data pointer member.

    op.getVarPtrMutable().assign(descriptor);
    op.setVarType(fir::unwrapRefType(descriptor.getType()));
    op.getMembersMutable().append(descriptorBaseAddrMembers);
    op.getBoundsMutable().assign(llvm::SmallVector<mlir::Value>{});

    if (auto mapClauseOwner =
            llvm::dyn_cast<mlir::omp::MapClauseOwningOpInterface>(target)) {
      llvm::SmallVector<mlir::Value> newMapOps;
      for (size_t i = 0; i < mapClauseOwner.getMapOperands().size(); ++i) {
        if (mapClauseOwner.getMapOperands()[i] == op) {
          // Push new implicit maps generated for the descriptor.
          newMapOps.push_back(descriptorBaseAddrMembers[0]);

          // for TargetOp's which have IsolatedFromAbove we must align the
          // new additional map operand with an appropriate BlockArgument,
          // as the printing and later processing currently requires a 1:1
          // mapping of BlockArgs to MapInfoOp's at the same placement in
          // each array (BlockArgs and MapOperands).
          if (auto targetOp = llvm::dyn_cast<mlir::omp::TargetOp>(target)) {
            targetOp.getRegion().insertArgument(
                i, descriptorBaseAddrMembers[0].getType(), loc);
          }

          newMapOps.push_back(mapClauseOwner.getMapOperands()[i]);
        } else {
          newMapOps.push_back(mapClauseOwner.getMapOperands()[i]);
        }
      }

      mapClauseOwner.getMapOperandsMutable().assign(newMapOps);
    }
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
          mlir::isa_and_present<fir::BoxAddrOp>(
              op.getVarPtr().getDefiningOp())) {
        builder.setInsertionPoint(op);
        // Currently a MapInfoOp argument can only show up on a single target
        // user so we can retrieve and use the first user.
        genDescriptorMemberMaps(op, builder, *op->getUsers().begin());
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
