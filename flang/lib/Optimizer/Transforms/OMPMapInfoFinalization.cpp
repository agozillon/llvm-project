//===- OMPMapInfoFinalization.cpp
//---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// \file
/// An OpenMP dialect related pass for FIR/HLFIR which performs some
/// pre-processing of MapInfoOp's after the module has been lowered to
/// finalize them.
///
/// For example, it expands MapInfoOp's containing descriptor related
/// types (fir::BoxType's) into multiple MapInfoOp's containing the parent
/// descriptor and pointer member components for individual mapping,
/// treating the descriptor type as a record type for later lowering in the
/// OpenMP dialect.
///
/// The pass also adds MapInfoOp's that are members of a parent object but are
/// not directly used in the body of a target region to its BlockArgument list
/// to maintain consistency across all MapInfoOp's tied to a region directly or
/// indirectly via a parent object.
//===----------------------------------------------------------------------===//

#include "flang/Lower/Support/Utils.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "flang/Semantics/runtime-type-info.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <map>
#include <numeric>

namespace fir {
#define GEN_PASS_DEF_OMPMAPINFOFINALIZATIONPASS
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

namespace {
class OMPMapInfoFinalizationPass
    : public fir::impl::OMPMapInfoFinalizationPassBase<
          OMPMapInfoFinalizationPass> {

  /// Small helper class tracking a members parent and its
  /// placement in the parents member list
  struct ParentAndPlacement {
    mlir::omp::MapInfoOp parent;
    size_t index;
  };

  /// Tracks any intermediate function/subroutine local allocations we
  /// generate for the descriptors of box type dummy arguments, so that
  /// we can retrieve it for subsequent reuses within the functions
  /// scope
  std::map</*descriptor opaque pointer=*/ void *,
           /*corresponding local alloca=*/ fir::AllocaOp>
      localBoxAllocas;

  /// getMemberUserList gathers all users of a particular MapInfoOp that are
  /// other MapInfoOp's and places them into the mapMemberUsers list, which
  /// records the map that the current argument MapInfoOp "op" is part of
  /// alongside the placement of "op" in the recorded users members list. The
  /// intent of the generated list is to find all MapInfoOp's that may be
  /// considered parents of the passed in "op" and in which it shows up in the
  /// member list, alongside collecting the placement information of "op" in its
  /// parents member list.
  void
  getMemberUserList(mlir::omp::MapInfoOp op,
                    llvm::SmallVectorImpl<ParentAndPlacement> &mapMemberUsers) {
    for (auto *users : op->getUsers())
      if (auto map = mlir::dyn_cast_if_present<mlir::omp::MapInfoOp>(users))
        for (size_t i = 0; i < map.getMembers().size(); ++i)
          if (map.getMembers()[i].getDefiningOp() == op)
            mapMemberUsers.push_back({map, i});
  }

  /// This function will expand a MapInfoOp's member indices back into a vector
  /// so that they can be trivially modified as unfortunately the attribute type
  /// that's used does not have modifiable fields at the moment (generally
  /// awkward to work with)
  void getMemberIndicesAsVectors(
      mlir::omp::MapInfoOp mapInfo,
      llvm::SmallVector<llvm::SmallVector<int32_t>> &indices) {
    size_t row = 0;
    size_t shapeX = mapInfo.getMembersIndexAttr().getShapedType().getShape()[0];
    size_t shapeJ = mapInfo.getMembersIndexAttr().getShapedType().getShape()[1];

    for (size_t i = 0; i < shapeX; ++i) {
      llvm::SmallVector<int32_t> vec;
      row = i * shapeJ;
      for (size_t j = 0; j < shapeJ; ++j) {
        vec.push_back(
            mapInfo.getMembersIndexAttr().getValues<int32_t>()[row + j]);
      }
      indices.push_back(vec);
    }
  }

  /// When provided a MapInfoOp containing a descriptor type that
  /// we must expand into multiple maps this function will extract
  /// the value from it and return it, in certain cases we must
  /// generate a new allocation to store into so that the
  /// fir::BoxOffsetOp we utilise to access the descriptor datas
  /// base address can be utilised.
  mlir::Value getDescriptorFromBoxMap(mlir::omp::MapInfoOp boxMap,
                                      fir::FirOpBuilder &builder) {
    mlir::Value descriptor = boxMap.getVarPtr();
    if (!fir::isTypeWithDescriptor(boxMap.getVarType()))
      if (auto addrOp = mlir::dyn_cast_if_present<fir::BoxAddrOp>(
              boxMap.getVarPtr().getDefiningOp()))
        descriptor = addrOp.getVal();

    // The fir::BoxOffsetOp only works with !fir.ref<!fir.box<...>> types, as
    // allowing it to access non-reference box operations can cause some
    // problematic SSA IR. However, in the case of assumed shape's the type
    // is not a !fir.ref, in these cases to retrieve the appropriate
    // !fir.ref<!fir.box<...>> to access the data we need to map we must
    // perform an alloca and then store to it and retrieve the data from the new
    // alloca.
    if (mlir::isa<fir::BaseBoxType>(descriptor.getType())) {
      mlir::Location loc = boxMap->getLoc();
      // if we have already created a local allocation for this BoxType,
      // we must be sure to re-use it so that we end up with the same allocations
      // being utilised for the same descriptor across all map uses, this prevents
      // runtime issues such as not appropriately releasing or deleting 
      // all mapped data.
      auto find = localBoxAllocas.find(descriptor.getAsOpaquePointer());
      if (find != localBoxAllocas.end()) {
        builder.create<fir::StoreOp>(loc, descriptor, find->second);
        descriptor = find->second;
      } else {
        mlir::OpBuilder::InsertPoint insPt = builder.saveInsertionPoint();
        mlir::Block *allocaBlock = builder.getAllocaBlock();
        assert(allocaBlock && "No alloca block found for this top level op");
        builder.setInsertionPointToStart(allocaBlock);
        auto alloca = builder.create<fir::AllocaOp>(loc, descriptor.getType());
        builder.restoreInsertionPoint(insPt);
        builder.create<fir::StoreOp>(loc, descriptor, alloca);
        localBoxAllocas[descriptor.getAsOpaquePointer()] = alloca;
        descriptor = alloca;
      }
    }

    return descriptor;
  }

  /// Simple function that will generate a FIR operation accessing
  /// the descriptors base address (BoxOffsetOp) and then generate a
  /// MapInfoOp for it, the most important thing to note is that
  /// we normally move the bounds from the descriptor map onto the
  /// base address map.
  mlir::omp::MapInfoOp getBaseAddrMap(mlir::Value descriptor,
                                      mlir::OperandRange bounds,
                                      int64_t mapType,
                                      fir::FirOpBuilder &builder) {
    mlir::Location loc = descriptor.getLoc();
    mlir::Value baseAddrAddr = builder.create<fir::BoxOffsetOp>(
        loc, descriptor, fir::BoxFieldAttr::base_addr);

    // Member of the descriptor pointing at the allocated data
    return builder.create<mlir::omp::MapInfoOp>(
        loc, baseAddrAddr.getType(), descriptor,
        mlir::TypeAttr::get(llvm::cast<mlir::omp::PointerLikeType>(
                                fir::unwrapRefType(baseAddrAddr.getType()))
                                .getElementType()),
        baseAddrAddr, mlir::SmallVector<mlir::Value>{},
        mlir::DenseIntElementsAttr{}, bounds,
        builder.getIntegerAttr(builder.getIntegerType(64, false), mapType),
        builder.getAttr<mlir::omp::VariableCaptureKindAttr>(
            mlir::omp::VariableCaptureKind::ByRef),
        builder.getStringAttr("") /*name*/,
        builder.getBoolAttr(false) /*partial_map*/);
  }

  /// This function generates MapInfoOp's for each of the relevant descriptor
  /// addendum components that are necessary. Currently this only maps the
  /// top level addendum data and the binding descriptor data, the rest of the
  /// relevant data components are left as a TODO for the moment when we
  /// encounter examples that require them (allows us the ability to check the
  /// components are appropriately being mapped).
  void createAddendumMaps(
      mlir::Value descriptor, mlir::SmallVector<mlir::Value> &newMembers,
      mlir::SmallVector<mlir::SmallVector<int32_t>> &memberIndices,
      fir::FirOpBuilder &builder) {
    auto module = descriptor.getDefiningOp()->getParentOfType<mlir::ModuleOp>();
    mlir::Location loc = descriptor.getLoc();

    // get access to the mapped record types type descriptor, a blob of global
    // constant data that is shared amongst identical types. For our case, we
    // primarily just need the type, we'll map the data via the descriptor which
    // holds a pointer to it. This only works for RecordTypes currently, which
    // may be the only case it needs to work, however, we make sure we
    // appropriately find a record type or emit an error as it's required for
    // further steps.
    fir::ClassType classType = mlir::dyn_cast<fir::ClassType>(
        fir::unwrapRefType(descriptor.getType()));
    mlir::Type declaredType = fir::getDerivedType(classType.getEleTy());
    fir::RecordType recTy = mlir::dyn_cast<fir::RecordType>(declaredType);
    mlir::Type typeDescTy;
    if (recTy) {
      std::string typeDescName =
          fir::NameUniquer::getTypeDescriptorName(recTy.getName());
      // FIXME/TODO: perhaps tag with declare target link Any, as it's a global
      // constant shared amongst all similar type instantiations, and is also
      // marked linkonce_odr.
      if (auto global = module.lookupSymbol<fir::GlobalOp>(typeDescName))
        typeDescTy = global.getType();
    } else {
      llvm_unreachable("unhandled addendum type");
    }

    // Access and map the addendum/derived type descriptor, the main blob of
    // data which contains a lot of other descriptors containing relevant
    // information about the original data stored in the descriptor type, things
    // like procedure function pointer bindings
    fir::RecordType typeDescRecTy = mlir::cast<fir::RecordType>(typeDescTy);
    mlir::Value addendumAddr = builder.create<fir::BoxOffsetOp>(
        loc, descriptor, fir::BoxFieldAttr::derived_type);

    memberIndices.push_back({7, -1, -1});
    newMembers.push_back(builder.create<mlir::omp::MapInfoOp>(
        loc, addendumAddr.getType(), descriptor,
        mlir::TypeAttr::get(typeDescRecTy), addendumAddr,
        mlir::SmallVector<mlir::Value>{}, mlir::DenseIntElementsAttr{},
        mlir::SmallVector<mlir::Value>{},
        builder.getIntegerAttr(
            builder.getIntegerType(64, false),
            static_cast<
                std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
                llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO)),
        builder.getAttr<mlir::omp::VariableCaptureKindAttr>(
            mlir::omp::VariableCaptureKind::ByRef),
        builder.getStringAttr("derived type descriptor") /*name*/,
        builder.getBoolAttr(false) /*partial_map*/));

    // Load the original data descriptor so we can appropriately access it's
    // type descriptor segment.
    mlir::Value loadedDesc = descriptor;
    if (fir::isa_ref_type(loadedDesc.getType()))
      loadedDesc = builder.create<fir::LoadOp>(loc, descriptor);

    // Load the type descriptor so we can access it's components to map the
    // relevant data.
    mlir::Type fieldTy = fir::FieldType::get(builder.getContext());
    mlir::Type tdescType =
        fir::TypeDescType::get(mlir::NoneType::get(builder.getContext()));
    mlir::Value boxDesc =
        builder.create<fir::BoxTypeDescOp>(loc, tdescType, loadedDesc);
    boxDesc = builder.create<fir::ConvertOp>(
        loc, fir::ReferenceType::get(typeDescTy), boxDesc);

    // Retrieve the bindings descriptor and access its base address to map the
    // attached data.
    const char *bindingsCompName = Fortran::semantics::bindingDescCompName;
    mlir::Value field = builder.create<fir::FieldIndexOp>(
        loc, fieldTy, bindingsCompName, typeDescRecTy, mlir::ValueRange{});
    mlir::Type coorTy =
        fir::ReferenceType::get(typeDescRecTy.getType(bindingsCompName));
    mlir::Value bindingBoxAddr =
        builder.create<fir::CoordinateOp>(loc, coorTy, boxDesc, field);
    mlir::Value bindingDataBaseAddr = builder.create<fir::BoxOffsetOp>(
        loc, bindingBoxAddr, fir::BoxFieldAttr::base_addr);

    // generate a map for the polymorphic bindings data.
    memberIndices.push_back({7, 0, 0});
    newMembers.push_back(builder.create<mlir::omp::MapInfoOp>(
        loc, bindingDataBaseAddr.getType(), descriptor,
        mlir::TypeAttr::get(
            llvm::cast<mlir::omp::PointerLikeType>(
                fir::unwrapRefType(bindingDataBaseAddr.getType()))
                .getElementType()),
        bindingDataBaseAddr, mlir::SmallVector<mlir::Value>{},
        mlir::DenseIntElementsAttr{}, mlir::SmallVector<mlir::Value>{},
        builder.getIntegerAttr(
            builder.getIntegerType(64, false),
            static_cast<
                std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
                llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO)),
        builder.getAttr<mlir::omp::VariableCaptureKindAttr>(
            mlir::omp::VariableCaptureKind::ByRef),
        builder.getStringAttr("polymorphic bindings") /*name*/,
        builder.getBoolAttr(false) /*partial_map*/));

    // TODO: Add further addendum component mappings as necessary.

    // Make sure to adjust all member indices to be of the same length
    // prepending -1s as necessary.
    Fortran::lower::omp::fillMemberIndices(memberIndices);
  }

  /// This function adjusts the member indices vector to include a new
  /// base address member, we take the position of the descriptor in
  /// the member indices list, which is the index data that the base
  /// addresses index will be based off of, as the base address is
  /// a member of the descriptor, we must also alter other members
  /// indices in the list to account for this new addition. This
  /// requires extending all members with -1's if the addition of
  /// the new base address has increased the member vector past the
  /// original size, as we must make sure all member indices are of
  /// the same length (think rectangle matrix) due to DenseIntElementsAttr
  /// requiring this. We also need to be aware that we are inserting
  /// into the middle of a member index vector in some cases (i.e.
  /// we could be accessing the member of a descriptor type with a
  /// subsequent map, so we must be sure to adjust any of these cases
  /// with the addition of the new base address index value).
  void adjustMemberIndices(
      llvm::SmallVector<llvm::SmallVector<int32_t>> &memberIndices,
      size_t memberIndex, size_t addrIndex) {
    // Find if the descriptor member we are basing our new base address index
    // off of has a -1 somewhere, indicating an empty index already exists (due
    // to a larger sized member position elsewhere) which allows us to simplify
    // later steps a little
    auto baseAddrIndex = memberIndices[memberIndex];
    auto *iterPos = std::find(baseAddrIndex.begin(), baseAddrIndex.end(), -1);

    // If we aren't at the end, as we found a -1, we can simply modify the -1
    // to the base addresses index in the descriptor (which will always be the
    // first member in the descriptor, so 0). If not, then we're extending the
    // index list and have to push on a 0 and adjust the position to the new
    // end.
    if (iterPos != baseAddrIndex.end()) {
      *iterPos = addrIndex;
    } else {
      baseAddrIndex.push_back(addrIndex);
      iterPos = baseAddrIndex.end();
    }

    auto isEqual = [](auto first1, auto last1, auto first2, auto last2) {
      int v1, v2;
      for (; first1 != last1; ++first1, ++first2) {
        v1 = (first1 == last1) ? -1 : *first1;
        v2 = (first2 == last2) ? -1 : *first2;

        if (!(v1 == v2))
          return false;
      }
      return true;
    };

    // If we find another member that is "derived/a member of" the descriptor
    // that is not the descriptor itself, we must insert a 0 for the new base
    // address we have just added for the descriptor into the list at the
    // appropriate position to maintain correctness of the positional/index data
    // for that member.
    size_t insertPosition = std::distance(baseAddrIndex.begin(), iterPos);
    for (size_t i = 0; i < memberIndices.size(); ++i) {
      if (isEqual(baseAddrIndex.begin(), iterPos, memberIndices[i].begin(),
                  memberIndices[i].end())) {
        if (i == memberIndex)
          continue;

        memberIndices[i].insert(
            std::next(memberIndices[i].begin(), insertPosition), addrIndex);
      }
    }

    // Insert our newly created baseAddrIndex into the larger list of indices at
    // the correct location.
    memberIndices.insert(std::next(memberIndices.begin(), memberIndex + 1),
                         baseAddrIndex);
  }

  // Adjusts the descriptors map type based on the map type of the original
  // map type which will apply to the base address, the main alteration that
  // is done currently is appending OMP_MAP_TO in cases where we only have
  // OMP_MAP_FROM or an ALLOC (lack of flag). This is because we will
  // always need to map the descriptor to device (or at the very least it
  // seems to be the case currently with the current lowered IR), as without
  // the appropriate descriptor information on the device there is a risk
  // of the kernel IR requesting for various data that will not have been
  // copied to perform things like indexing, this can cause segfaults and
  // memory access errors. These alterations are only unapplicable to
  // target exit currently.
  unsigned long getDescriptorMapType(unsigned long mapTypeFlag,
                                     mlir::Operation *target) {
    auto newDescFlag = llvm::omp::OpenMPOffloadMappingFlags(mapTypeFlag);

    if ((llvm::isa_and_nonnull<mlir::omp::TargetDataOp>(target) ||
         llvm::isa_and_nonnull<mlir::omp::TargetOp>(target)) &&
        static_cast<
            std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
            (newDescFlag &
             llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM)) &&
        static_cast<
            std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
            (newDescFlag & llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO) !=
            llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO))
      return static_cast<
          std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
          newDescFlag | llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO);

    if ((llvm::isa_and_nonnull<mlir::omp::TargetDataOp>(target) ||
         llvm::isa_and_nonnull<mlir::omp::TargetEnterDataOp>(target)) &&
        mapTypeFlag == 0)
      return static_cast<
          std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
          llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO);

    return mapTypeFlag;
  }

  mlir::omp::MapInfoOp genDescriptorMemberMaps(mlir::omp::MapInfoOp op,
                                               fir::FirOpBuilder &builder,
                                               mlir::Operation *target) {
    llvm::SmallVector<ParentAndPlacement> mapMemberUsers;
    getMemberUserList(op, mapMemberUsers);

    // TODO: map the addendum segment of the descriptor, similarly to the
    // base address/data pointer member.
    mlir::Value descriptor = getDescriptorFromBoxMap(op, builder);
    auto baseAddr = getBaseAddrMap(descriptor, op.getBounds(),
                                   op.getMapType().value(), builder);

    mlir::DenseIntElementsAttr newMembersAttr;
    mlir::SmallVector<mlir::Value> newMembers;
    llvm::SmallVector<llvm::SmallVector<int32_t>> memberIndices;
    if (!mapMemberUsers.empty() || !op.getMembers().empty())
      getMemberIndicesAsVectors(
          !mapMemberUsers.empty() ? mapMemberUsers[0].parent : op,
          memberIndices);

    // If the operation that we are expanding with a descriptor has a user
    // (parent), then we have to expand the parents member indices to reflect
    // the adjusted member indices for the base address insertion. However, if
    // it does not then we are expanding a MapInfoOp without any pre-existing
    // member information to now have one new member for the base address or we
    // are expanding a parent that is a descriptor and we have to adjust all of
    // it's members to reflect the insertion of the base address.
    if (!mapMemberUsers.empty()) {
      adjustMemberIndices(memberIndices, mapMemberUsers[0].index, 0);
      llvm::SmallVector<mlir::Value> newMemberOps;
      mlir::OperandRange membersArr = mapMemberUsers[0].parent.getMembers();
      for (size_t i = 0; i < membersArr.size(); ++i) {
        newMemberOps.push_back(membersArr[i]);
        if (membersArr[i] == op)
          newMemberOps.push_back(baseAddr);
      }
      mapMemberUsers[0].parent.getMembersMutable().assign(newMemberOps);
      Fortran::lower::omp::fillMemberIndices(memberIndices);
      mapMemberUsers[0].parent.setMembersIndexAttr(
          Fortran::lower::omp::createDenseElementsAttrFromIndices(memberIndices,
                                                                  builder));
    } else {
      newMembers.push_back(baseAddr);
      if (!op.getMembers().empty() || !memberIndices.empty()) {
        for (auto &indices : memberIndices)
          indices.insert(indices.begin(), 0);
        llvm::SmallVector<int> baseAddrIndex;
        baseAddrIndex.resize(memberIndices[0].size());
        std::fill(baseAddrIndex.begin(), baseAddrIndex.end(), -1);
        baseAddrIndex[0] = 0;
        memberIndices.insert(memberIndices.begin(), baseAddrIndex);
        Fortran::lower::omp::fillMemberIndices(memberIndices);

        newMembersAttr =
            Fortran::lower::omp::createDenseElementsAttrFromIndices(
                memberIndices, builder);
        newMembers.append(op.getMembers().begin(), op.getMembers().end());
      } else {
        memberIndices.push_back({0});
        newMembersAttr =
            Fortran::lower::omp::createDenseElementsAttrFromIndices(
                memberIndices, builder);
      }
    }

   fir::BaseBoxType boxType =
        mlir::cast<fir::BaseBoxType>(fir::unwrapRefType(descriptor.getType()));
    if (fir::boxHasAddendum(boxType))
      createAddendumMaps(descriptor, newMembers, memberIndices, builder);

    mlir::omp::MapInfoOp newDescParentMapOp =
        builder.create<mlir::omp::MapInfoOp>(
            op->getLoc(), op.getResult().getType(), descriptor,
            mlir::TypeAttr::get(fir::unwrapRefType(descriptor.getType())),
            mlir::Value{}, newMembers, newMembersAttr /*members_index*/,
            mlir::SmallVector<mlir::Value>{},
            builder.getIntegerAttr(
                builder.getIntegerType(64, false),
                getDescriptorMapType(op.getMapType().value(), target)),
            op.getMapCaptureTypeAttr(), op.getNameAttr(),
            builder.getBoolAttr(false));
    op.replaceAllUsesWith(newDescParentMapOp.getResult());
    op->erase();
    return newDescParentMapOp;
  }

  // We add all mapped record members not directly used in the target region
  // to the block arguments in front of their parent and we place them into
  // the map operands list for consistency.
  //
  // These indirect uses (via accesses to their parent) will still be
  // mapped individually in most cases, and a parent mapping doesn't
  // guarantee the parent will be mapped in its totality, partial
  // mapping is common.
  //
  // For example:
  //    map(tofrom: x%y)
  //
  // Will generate a mapping for "x" (the parent) and "y" (the member).
  // The parent "x" will not be mapped, but the member "y" will.
  // However, we must have the parent as a BlockArg and MapOperand
  // in these cases, to maintain the correct uses within the region and
  // to help tracking that the member is part of a larger object.
  //
  // In the case of:
  //    map(tofrom: x%y, x%z)
  //
  // The parent member becomes more critical, as we perform a partial
  // structure mapping where we link the mapping of the members y
  // and z together via the parent x. We do this at a kernel argument
  // level in LLVM IR and not just MLIR, which is important to maintain
  // similarity to Clang and for the runtime to do the correct thing.
  // However, we still do not map the structure in its totality but
  // rather we generate an un-sized "binding" map entry for it.
  //
  // In the case of:
  //    map(tofrom: x, x%y, x%z)
  //
  // We do actually map the entirety of "x", so the explicit mapping of
  // x%y, x%z becomes unnecessary. It is redundant to write this from a
  // Fortran OpenMP perspective (although it is legal), as even if the
  // members were allocatables or pointers, we are mandated by the
  // specification to map these (and any recursive components) in their
  // entirety, which is different to the C++ equivalent, which requires
  // explicit mapping of these segments.
  void addImplicitMembersToTarget(mlir::omp::MapInfoOp op,
                                  fir::FirOpBuilder &builder,
                                  mlir::Operation *target) {
    auto mapClauseOwner =
        llvm::dyn_cast_if_present<mlir::omp::MapClauseOwningOpInterface>(
            target);
    if (!mapClauseOwner)
      return;

    llvm::SmallVector<mlir::Value> newMapOps;
    mlir::OperandRange mapOperandsArr = mapClauseOwner.getMapOperands();
    auto targetOp = llvm::dyn_cast<mlir::omp::TargetOp>(target);

    for (size_t i = 0; i < mapOperandsArr.size(); ++i) {
      if (mapOperandsArr[i] == op) {
        for (auto [j, mapMember] : llvm::enumerate(op.getMembers())) {
          newMapOps.push_back(mapMember);
          // for TargetOp's which have IsolatedFromAbove we must align the
          // new additional map operand with an appropriate BlockArgument,
          // as the printing and later processing currently requires a 1:1
          // mapping of BlockArgs to MapInfoOp's at the same placement in
          // each array (BlockArgs and MapOperands).
          if (targetOp) {
            targetOp.getRegion().insertArgument(i + j, mapMember.getType(),
                                                targetOp->getLoc());
          }
        }
      }
      newMapOps.push_back(mapOperandsArr[i]);
    }
    mapClauseOwner.getMapOperandsMutable().assign(newMapOps);
  }

  // We retrieve the first user that is a Target operation, there
  // should only be one currently, every MapInfoOp can be tied to
  // at most 1 Target operation and at the minimum no operation,
  // this may change in the future with IR cleanups/modifications
  // in which case this pass will need updated to support cases
  // where a map can have more than one user and more than one of
  // those users can be a Target operation. For now, we simply
  // return the first target operation encountered, which may
  // be on the parent MapInfoOp in the case of a member mapping
  // in which case we must traverse the MapInfoOp chain until we
  // find the first TargetOp user.
  mlir::Operation *getFirstTargetUser(mlir::omp::MapInfoOp mapOp) {
    for (auto *user : mapOp->getUsers()) {
      if (llvm::isa<mlir::omp::TargetOp>(user) ||
          llvm::isa<mlir::omp::TargetDataOp>(user) ||
          llvm::isa<mlir::omp::TargetEnterDataOp>(user) ||
          llvm::isa<mlir::omp::TargetExitDataOp>(user) ||
          llvm::isa<mlir::omp::TargetUpdateOp>(user))
        return user;

      if (auto mapUser = llvm::dyn_cast_if_present<mlir::omp::MapInfoOp>(user))
        return getFirstTargetUser(mapUser);
    }

    return nullptr;
  }

  // This pass executes on omp::MapInfoOp's containing descriptor based types
  // (allocatables, pointers, assumed shape etc.) and expanding them into
  // multiple omp::MapInfoOp's for each pointer member contained within the
  // descriptor.
  //
  // From the perspective of the MLIR pass manager this runs on the top level
  // operation (usually function) containing the MapInfoOp because this pass
  // will mutate siblings of MapInfoOp.
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::ModuleOp module = func->getParentOfType<mlir::ModuleOp>();
    fir::KindMapping kindMap = fir::getKindMapping(module);
    fir::FirOpBuilder builder{module, std::move(kindMap)};

    func->walk([&](mlir::omp::MapInfoOp op) {
      // TODO: Currently only supports a single user for the MapInfoOp, this
      // is fine for the moment as the Fortran frontend will generate a
      // new MapInfoOp with at most one user currently, in the case of
      // members of other objects like derived types, the user would be the
      // parent, in cases where it's a regular non-member map the user would
      // be the target operation it is being mapped by.
      //
      // However, when/if we optimise/cleanup the IR we will have to extend
      // this pass to support multiple users, as I would imagine we may wish
      // to have a map be re-used by multiple users (e.g. across multiple
      // targets that map the variable and have identical map properties)
      assert(llvm::hasSingleElement(op->getUsers()) &&
             "OMPMapInfoFinalization currently only supports single users "
             "of a MapInfoOp");

      // may have to find the hlfir.declareop for it and find out what it's spawned from.
      // e.g. if it's original variable is a box type or some other type...
      if (fir::isTypeWithDescriptor(op.getVarType()) ||
          mlir::isa_and_present<fir::BoxAddrOp>(
              op.getVarPtr().getDefiningOp())) {
        builder.setInsertionPoint(op);
        genDescriptorMemberMaps(op, builder, getFirstTargetUser(op));
      }
    });

    // Wait until after we have generated all of our maps to add them onto
    // the targets block arguments, simplifying the process as no need to
    // avoid accidental duplicate additions
    func->walk([&](mlir::omp::MapInfoOp op) {
      addImplicitMembersToTarget(op, builder, getFirstTargetUser(op));
    });
  }
};

} // namespace
