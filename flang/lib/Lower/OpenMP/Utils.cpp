//===-- Utils..cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "Utils.h"
#include "Clauses.h"

#include <cstdint>
#include <numeric>
#include <flang/Lower/AbstractConverter.h>
#include <flang/Lower/ConvertType.h>
#include <flang/Optimizer/Builder/FIRBuilder.h>
#include <flang/Parser/parse-tree.h>
#include <flang/Parser/tools.h>
#include <flang/Semantics/tools.h>
#include <llvm/Support/CommandLine.h>

llvm::cl::opt<bool> treatIndexAsSection(
    "openmp-treat-index-as-section",
    llvm::cl::desc("In the OpenMP data clauses treat `a(N)` as `a(N:N)`."),
    llvm::cl::init(true));

llvm::cl::opt<bool> enableDelayedPrivatization(
    "openmp-enable-delayed-privatization",
    llvm::cl::desc(
        "Emit `[first]private` variables as clauses on the MLIR ops."),
    llvm::cl::init(false));

namespace Fortran {
namespace lower {
namespace omp {

void genObjectList(const ObjectList &objects,
                   Fortran::lower::AbstractConverter &converter,
                   llvm::SmallVectorImpl<mlir::Value> &operands) {
  for (const Object &object : objects) {
    const Fortran::semantics::Symbol *sym = object.id();
    assert(sym && "Expected Symbol");
    if (mlir::Value variable = converter.getSymbolAddress(*sym)) {
      operands.push_back(variable);
    } else if (const auto *details =
                   sym->detailsIf<Fortran::semantics::HostAssocDetails>()) {
      operands.push_back(converter.getSymbolAddress(details->symbol()));
      converter.copySymbolBinding(details->symbol(), *sym);
    }
  }
}

void genObjectList2(const Fortran::parser::OmpObjectList &objectList,
                    Fortran::lower::AbstractConverter &converter,
                    llvm::SmallVectorImpl<mlir::Value> &operands) {
  auto addOperands = [&](Fortran::lower::SymbolRef sym) {
    const mlir::Value variable = converter.getSymbolAddress(sym);
    if (variable) {
      operands.push_back(variable);
    } else if (const auto *details =
                   sym->detailsIf<Fortran::semantics::HostAssocDetails>()) {
      operands.push_back(converter.getSymbolAddress(details->symbol()));
      converter.copySymbolBinding(details->symbol(), sym);
    }
  };
  for (const Fortran::parser::OmpObject &ompObject : objectList.v) {
    Fortran::semantics::Symbol *sym = getOmpObjectSymbol(ompObject);
    addOperands(*sym);
  }
}

void gatherFuncAndVarSyms(
    const ObjectList &objects, mlir::omp::DeclareTargetCaptureClause clause,
    llvm::SmallVectorImpl<DeclareTargetCapturePair> &symbolAndClause) {
  for (const Object &object : objects)
    symbolAndClause.emplace_back(clause, *object.id());
}

void checkAndApplyDeclTargetMapFlags(
    Fortran::lower::AbstractConverter &converter,
    llvm::omp::OpenMPOffloadMappingFlags &mapFlags,
    const Fortran::semantics::Symbol &symbol) {
  if (auto declareTargetOp =
          llvm::dyn_cast_if_present<mlir::omp::DeclareTargetInterface>(
              converter.getModuleOp().lookupSymbol(
                  converter.mangleName(symbol)))) {
    // Only Link clauses have OMP_MAP_PTR_AND_OBJ applied, To clause
    // seems to function differently.
    if (declareTargetOp.getDeclareTargetCaptureClause() ==
        mlir::omp::DeclareTargetCaptureClause::link)
      mapFlags |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_PTR_AND_OBJ;
  }
}

int findComponentMemberPlacement(
    const Fortran::semantics::Symbol *dTypeSym,
    const Fortran::semantics::Symbol *componentSym) {
  const auto *derived =
      dTypeSym->detailsIf<Fortran::semantics::DerivedTypeDetails>();
  
  int placement = 0;
  for (auto t : derived->componentNames()) {
    if (t == componentSym->name())
      return placement;
    placement++;
  }

  return -1;
}

const parser::StructureComponent *getStructComp(const parser::DataRef &x) {
  const parser::StructureComponent* comp = nullptr;
   common::visit(
      common::visitors{
          [&](const parser::Name &name)  { comp = nullptr; },
          [&](const common::Indirection<parser::StructureComponent> &sc) { comp = &sc.value(); },
          [&](const common::Indirection<parser::ArrayElement> &ae) { comp = getStructComp(ae.value().base); 
            },
          [&](const common::Indirection<parser::CoindexedNamedObject> &ci) { comp = getStructComp(ci.value().base); },
      },
      x.u);

  return comp;
}

const parser::StructureComponent *getStructComp(const parser::Substring &x) {
  return getStructComp(std::get<parser::DataRef>(x.t));
}

const parser::StructureComponent *getStructComp(const parser::Designator &x) {
  const parser::StructureComponent *comp = nullptr;
  common::visit(common::visitors{
                    [&](const auto &y) { comp = getStructComp(y); },
                },
                x.u);
  return comp;
}

int getComponentPlacementInParent(
    const Fortran::semantics::Symbol *componentSym) {
  const auto *derived =
      componentSym->owner()
          .derivedTypeSpec()
          ->typeSymbol()
          .detailsIf<Fortran::semantics::DerivedTypeDetails>();

  int placement = 0;
  for (auto t : derived->componentNames()) {
    if (t == componentSym->name())
      return placement;
    placement++;
  }

  return -1;
}

llvm::SmallVector<int>
generateMemberPlacementIndices(const Fortran::parser::OmpObject &ompObject) {
  assert(getOmpObjectSymbol(ompObject)->owner().IsDerivedType() &&
         "Expected an OmpObject that was a component of a derived type");
  const auto *designator =
      Fortran::parser::Unwrap<Fortran::parser::Designator>(ompObject.u);
  assert(designator && "Expected a designator from derived type "
                       "component during map clause processing");
  const Fortran::parser::StructureComponent *curComp =
      getStructComp(*designator);

  std::list<int> indices;
  while (curComp) {
    indices.push_front(
        getComponentPlacementInParent(curComp->component.symbol));
    curComp = getStructComp(curComp->base);
  }

  return llvm::SmallVector<int>{std::begin(indices), std::end(indices)};
}

mlir::DenseIntElementsAttr createDenseElementsAttrFromIndices(
    llvm::SmallVector<llvm::SmallVector<int>> &memberPlacementIndices,
    fir::FirOpBuilder &builder) {
  llvm::SmallVector<int64_t> shape;
  for (auto v : memberPlacementIndices)
    shape.push_back(v.size());

  llvm::SmallVector<int> indicesFlattened =
      std::accumulate(memberPlacementIndices.begin(),
                      memberPlacementIndices.end(), llvm::SmallVector<int>(),
                      [](llvm::SmallVector<int> &x, llvm::SmallVector<int> &y) {
                        x.insert(x.end(), y.begin(), y.end());
                        return x;
                      });

  return mlir::DenseIntElementsAttr::get(
      mlir::VectorType::get(llvm::ArrayRef<int64_t>(shape),
                            builder.getIntegerType(64)),
      indicesFlattened);
}

void insertChildMapInfoIntoParent(
    Fortran::lower::AbstractConverter &converter,
    llvm::SmallVector<const Fortran::semantics::Symbol *> &memberParentSyms,
    llvm::SmallVector<mlir::omp::MapInfoOp> &memberMaps,
    llvm::SmallVector<llvm::SmallVector<int>> &memberPlacementIndices,
    llvm::SmallVectorImpl<mlir::Value> &mapOperands,
    llvm::SmallVectorImpl<mlir::Type> *mapSymTypes,
    llvm::SmallVectorImpl<mlir::Location> *mapSymLocs,
    llvm::SmallVectorImpl<const Fortran::semantics::Symbol *> *mapSymbols) {
  for (auto [idx, sym] : llvm::enumerate(memberParentSyms)) {
    bool parentExists = false;
    size_t parentIdx = 0;
    for (size_t i = 0; i < mapSymbols->size(); ++i) {
      if ((*mapSymbols)[i] == sym) {
        parentExists = true;
        parentIdx = i;
        break;
      }
    }

I think we will have to change the below, to create or append the mapinfoop at the end, and we 
divide up the members indices between them all beforehand so we have to create the denseattr a minimal
number of times. perhaps tracking a std::pair of map To member indices...
    if (parentExists) {
      auto mapOp = mlir::dyn_cast<mlir::omp::MapInfoOp>(
          mapOperands[parentIdx].getDefiningOp());
      assert(mapOp && "Parent provided to insertChildMapInfoIntoParent was not "
                      "an expected MapInfoOp");



      // found a parent, append.
      mapOp.getMembersMutable().append((mlir::Value)memberMaps[idx]);

// this very likely won't work the same anymore... as it's no longer a single 1-D index
        // mlir::DenseIntElementsAttr::get(
        //     mlir::VectorType::get(llvm::ArrayRef<int64_t>(lShape),
        //                           IntegerType::get(module.getContext(), 64)),
        //     llvm::ArrayRef<int64_t>{lBounds})

      // llvm::SmallVector<mlir::Attribute> memberIndexTmp{
      //     mapOp.getMembersIndexAttr().begin(),
      //     mapOp.getMembersIndexAttr().end()};
      // memberIndexTmp.push_back(memberPlacementIndices[idx]);
      // mapOp.setMembersIndexAttr(mlir::DenseIntElementsAttr::get(
      //     /*converter.getFirOpBuilder().getContext(), memberIndexTmp)*/);
    } else {
      // NOTE: We take the map type of the first child, this may not
      // be the correct thing to do, however, we shall see. For the moment
      // it allows this to work with enter and exit without causing MLIR
      // verification issues. The more appropriate thing may be to take
      // the "main" map type clause from the directive being used.
      uint64_t mapType = memberMaps[idx].getMapType().value_or(0);

      // create parent to emplace and bind members
      auto origSymbol = converter.getSymbolAddress(*sym);
      mlir::Value mapOp = createMapInfoOp(
          converter.getFirOpBuilder(), origSymbol.getLoc(), origSymbol,
          mlir::Value(), sym->name().ToString(), {}, {memberMaps[idx]},
          mlir::ArrayAttr{}/*mlir::ArrayAttr::get(converter.getFirOpBuilder().getContext(),
                               memberPlacementIndices[idx])*/,
          mapType, mlir::omp::VariableCaptureKind::ByRef, origSymbol.getType(),
          true);

      mapOperands.push_back(mapOp);
      if (mapSymTypes)
        mapSymTypes->push_back(mapOp.getType());
      if (mapSymLocs)
        mapSymLocs->push_back(mapOp.getLoc());
      if (mapSymbols)
        mapSymbols->push_back(sym);
    }
  }
}

Fortran::semantics::Symbol *
getOmpObjectSymbol(const Fortran::parser::OmpObject &ompObject) {
  Fortran::semantics::Symbol *sym = nullptr;
  std::visit(
      Fortran::common::visitors{
          [&](const Fortran::parser::Designator &designator) {
            if (auto *arrayEle =
                    Fortran::parser::Unwrap<Fortran::parser::ArrayElement>(
                        designator)) {
              sym = GetLastName(arrayEle->base).symbol;
            } else if (auto *structComp = Fortran::parser::Unwrap<
                           Fortran::parser::StructureComponent>(designator)) {
              sym = structComp->component.symbol;
            } else if (const Fortran::parser::Name *name =
                           Fortran::semantics::getDesignatorNameIfDataRef(
                               designator)) {
              sym = name->symbol;
            }
          },
          [&](const Fortran::parser::Name &name) { sym = name.symbol; }},
      ompObject.u);
  return sym;
}

} // namespace omp
} // namespace lower
} // namespace Fortran
