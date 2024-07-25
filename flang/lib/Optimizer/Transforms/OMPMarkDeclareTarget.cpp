#include "flang/Optimizer/Support/Utils.h"
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
#define GEN_PASS_DEF_OMPMARKDECLARETARGETPASS
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

namespace {
class OMPMarkDeclareTargetPass
    : public fir::impl::OMPMarkDeclareTargetPassBase<OMPMarkDeclareTargetPass> {

  void markNestedFuncs(mlir::omp::DeclareTargetDeviceType parentDevTy,
                       mlir::omp::DeclareTargetCaptureClause parentCapClause,
                       mlir::Operation *currOp,
                       llvm::SmallPtrSet<mlir::Operation *, 16> visited,
                       const fir::BindingTables &bindingTables) {
    if (visited.contains(currOp))
      return;
    visited.insert(currOp);

    auto markFuncOp = [&](mlir::func::FuncOp currFOp) {
      auto current = llvm::dyn_cast<mlir::omp::DeclareTargetInterface>(
          currFOp.getOperation());

      if (current.isDeclareTarget()) {
        auto currentDt = current.getDeclareTargetDeviceType();

        // Found the same function twice, with different device_types,
        // mark as Any as it belongs to both
        if (currentDt != parentDevTy &&
            currentDt != mlir::omp::DeclareTargetDeviceType::any) {
          current.setDeclareTarget(mlir::omp::DeclareTargetDeviceType::any,
                                   current.getDeclareTargetCaptureClause());
        }
      } else
        current.setDeclareTarget(parentDevTy, parentCapClause);

      markNestedFuncs(parentDevTy, parentCapClause, currFOp, visited,
                      bindingTables);
    };

    currOp->walk([&, this](mlir::Operation *op) {
      if (auto callOp = llvm::dyn_cast<mlir::CallOpInterface>(op)) {
        if (auto symRef = llvm::dyn_cast_if_present<mlir::SymbolRefAttr>(
                callOp.getCallableForCallee()))
          if (auto currFOp =
                  getOperation().lookupSymbol<mlir::func::FuncOp>(symRef))
            markFuncOp(currFOp);
      }
    });

    auto findMatchingMethods = [&](llvm::StringRef methodName) {
      llvm::SmallVector<mlir::func::FuncOp> result;

      for (auto &[tableName, table] : bindingTables) {
        for (auto &[tableEntry, bindingInfo] : table) {
          if (tableEntry != methodName)
            continue;

          if (auto currFOp = getOperation().lookupSymbol<mlir::func::FuncOp>(
                  bindingInfo.proc))
            result.push_back(currFOp);
        }
      }

      return result;
    };

    currOp->walk([&](fir::DispatchOp dispatchOp) {
      llvm::SmallVector<mlir::func::FuncOp> methods =
          findMatchingMethods(dispatchOp.getMethod());

      for (mlir::func::FuncOp method : methods)
        markFuncOp(method);
    });
  }

  // This pass executes on mlir::ModuleOp's marking functions contained within
  // as implicitly declare target if they are called from within an explicitly
  // marked declare target function or a target region (TargetOp)
  void runOnOperation() override {
    fir::BindingTables bindingTables;
    fir::buildBindingTables(bindingTables, getOperation());

    for (auto functionOp : getOperation().getOps<mlir::func::FuncOp>()) {
      auto declareTargetOp = llvm::dyn_cast<mlir::omp::DeclareTargetInterface>(
          functionOp.getOperation());
      if (declareTargetOp.isDeclareTarget()) {
        llvm::SmallPtrSet<mlir::Operation *, 16> visited;
        markNestedFuncs(declareTargetOp.getDeclareTargetDeviceType(),
                        declareTargetOp.getDeclareTargetCaptureClause(),
                        functionOp, visited, bindingTables);
      }
    }

    // TODO: Extend to work with reverse-offloading, this shouldn't
    // require too much effort, just need to check the device clause
    // when it's lowering has been implemented and change the
    // DeclareTargetDeviceType argument from nohost to host depending on
    // the contents of the device clause
    getOperation()->walk([&](mlir::omp::TargetOp tarOp) {
      llvm::SmallPtrSet<mlir::Operation *, 16> visited;
      markNestedFuncs(mlir::omp::DeclareTargetDeviceType::nohost,
                      mlir::omp::DeclareTargetCaptureClause::to, tarOp, visited,
                      bindingTables);
    });
  }
};

} // namespace
