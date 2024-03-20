//===-- Lower/OpenMP/Utils.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_OPENMPUTILS_H
#define FORTRAN_LOWER_OPENMPUTILS_H

#include "Clauses.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/CommandLine.h"
#include <list>

extern llvm::cl::opt<bool> treatIndexAsSection;
extern llvm::cl::opt<bool> enableDelayedPrivatization;

namespace fir {
class FirOpBuilder;
} // namespace fir

namespace llvm::omp {
enum class OpenMPOffloadMappingFlags : uint64_t;
} // namespace llvm::omp

namespace Fortran {

namespace semantics {
class Symbol;
} // namespace semantics

namespace parser {
struct OmpObject;
struct OmpObjectList;
struct Designator;
} // namespace parser

namespace lower {

class AbstractConverter;

namespace omp {

using DeclareTargetCapturePair =
    std::pair<mlir::omp::DeclareTargetCaptureClause,
              const Fortran::semantics::Symbol &>;

llvm::SmallVector<int>
generateMemberPlacementIndices(const Fortran::parser::OmpObject &ompObject);

mlir::omp::MapInfoOp
createMapInfoOp(fir::FirOpBuilder &builder, mlir::Location loc,
                mlir::Value baseAddr, mlir::Value varPtrPtr, std::string name,
                mlir::SmallVector<mlir::Value> bounds,
                mlir::SmallVector<mlir::Value> members,
                mlir::DenseIntElementsAttr membersIndex, uint64_t mapType,
                mlir::omp::VariableCaptureKind mapCaptureType, mlir::Type retTy,
                bool partialMap = false);

int findComponentMemberPlacement(
    const Fortran::semantics::Symbol *dTypeSym,
    const Fortran::semantics::Symbol *componentSym);

void insertChildMapInfoIntoParent(
    Fortran::lower::AbstractConverter &converter,
    std::map<const Fortran::semantics::Symbol *,
             llvm::SmallVector<std::pair<llvm::SmallVector<int>, int>>>
        &parentMemberIndices,
    llvm::SmallVector<mlir::omp::MapInfoOp> &memberMaps,
    llvm::SmallVectorImpl<mlir::Value> &mapOperands,
    llvm::SmallVectorImpl<mlir::Type> *mapSymTypes,
    llvm::SmallVectorImpl<mlir::Location> *mapSymLocs,
    llvm::SmallVectorImpl<const Fortran::semantics::Symbol *> *mapSymbols);

void gatherFuncAndVarSyms(
    const ObjectList &objects, mlir::omp::DeclareTargetCaptureClause clause,
    llvm::SmallVectorImpl<DeclareTargetCapturePair> &symbolAndClause);

const parser::StructureComponent *getStructComp(const parser::DataRef &x);

const parser::StructureComponent *getStructComp(const parser::Substring &x);

const parser::StructureComponent *getStructComp(const parser::Designator &x);

Fortran::semantics::Symbol *
getOmpObjectSymbol(const Fortran::parser::OmpObject &ompObject);

void genObjectList(const ObjectList &objects,
                   Fortran::lower::AbstractConverter &converter,
                   llvm::SmallVectorImpl<mlir::Value> &operands);

void genObjectList2(const Fortran::parser::OmpObjectList &objectList,
                    Fortran::lower::AbstractConverter &converter,
                    llvm::SmallVectorImpl<mlir::Value> &operands);

} // namespace omp
} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_OPENMPUTILS_H
