<!--===- docs/OpenMP-descriptor-management.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# OpenMP dialect: Fortran descriptor type mapping for offload

The descriptor mapping for OpenMP currently works differently to the planned direction for OpenACC, however, 
it is possible and would likely be ideal to align the method with OpenACC in the future. However, at least 
currently the OpenMP specification is less descriptive and has less stringent rules around descriptor based
types so does not require as complex a set of descriptor management rules (although, in certain cases 
for the interim adopting OpenACC's rules where it makes sense could be useful).

The initial method for mapping Fortran types tied to descriptors for OpenMP offloading is to treat these types 
as a special case of OpenMP record type (C/C++ structure/class, Fortran derived type etc.) mapping as far as the 
runtime is concerned. Where the box (descriptor information) is the holding container and the underlying 
data pointer is contained within the container, and we must generate explicit maps for both the pointer member and
the container. As an example, a small C++ program that is equivalent to the concept described:

```C++
struct mock_descriptor {
  long int x;
  std::byte x1, x2, x3, x4;
  void *pointer;
  long int lx[1][3];
};

int main() {
mock_descriptor data;
#pragma omp target map(tofrom: data, data.pointer[:upper_bound])
{
    do something... 
}

 return 0;
}
```

In the above, we have to map both the containing structure, with its non-pointer members and the
data pointed to by the pointer contained within the structure to appropriately access the data. This 
is effectively what is done with descriptor types for the time being. Other pointers that are part 
of the descriptor container such as the addendum should also be treated as the data pointer is 
treated.

Currently, Flang will lower these descriptor types in the OpenMP lowering (lower/OpenMP.cpp) similarly
to all other map types, generating an omp.MapInfoOp containing relevant information required for lowering
the OpenMP dialect to LLVM-IR during the final stages of the MLIR lowering. However, after 
the lowering to FIR/HLFIR has been performed an OpenMP dialect specific pass for Fortran, 
OMPDescriptorMapInfoGenPass (Optimizer/OMPDescriptorMapInfoGen.cpp) will expand the 
omp.MapInfoOp's containing descriptors (which currently will be a BoxType or BoxAddrOp) into multiple 
mappings, with one extra per pointer member in the descriptor that is supported on top of the original
descriptor map operation. These pointers members are linked to the parent descriptor by adding them to 
the member field of the original descriptor map operation, they are then inserted into the relevant map
owning operation's (omp.TargetOp, omp.DataOp etc.) map operand list and in cases where the owning operation
is IsolatedFromAbove, it also inserts them as BlockArgs to canonicalize the mappings and simplify lowering.

An example transformation by the OMPDescriptorMapInfoGenPass:

```

...
%12 = omp.map_info var_ptr(%1#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.box<!fir.ptr<!fir.array<?xi32>>>) map_clauses(tofrom) capture(ByRef) bounds(%11) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {name = "arg_alloc"}
...
omp.target map_entries(%12 -> %arg1, %13 -> %arg2 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<i32>) {
    ^bb0(%arg1: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, %arg2: !fir.ref<i32>):
...

====>

...
%12 = fir.box_offset %1#1 base_addr : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
%13 = omp.map_info var_ptr(%12 : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, !fir.array<?xi32>) map_clauses(tofrom) capture(ByRef) bounds(%11) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {name = ""}
%14 = omp.map_info var_ptr(%1#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.box<!fir.ptr<!fir.array<?xi32>>>) map_clauses(tofrom) capture(ByRef) members(%13 : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {name = "arg_alloc"}
...
omp.target map_entries(%13 -> %arg1, %14 -> %arg2, %15 -> %arg3 : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<i32>) {
    ^bb0(%arg1: !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, %arg2: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, %arg3: !fir.ref<i32>):
...

```

In later stages of the compilation flow when the OpenMP dialect is being lowered to LLVM-IR these descriptor
mappings are treated as if they were structure mappings with explicit member maps on the same directive as 
their parent was mapped.

This method is generic in the sense that the OpenMP diaelct doesn't need to understand that it is mapping a 
Fortran type containing a descriptor, it just thinks it's a record type from either Fortran or C++. However,
it is a little rigid in how the descriptor mappings are handled as there is no specialisation or possibility
to specialise the mappings for possible edge cases without poluting the dialect or lowering with further
knowledge of Fortran and the FIR dialect. In the case that this kind of specialisation is required or 
desired then the methodology described by OpenACC which utilises runtime functions to handle specialised mappings
for dialects may be a more desirable approach to move towards. For the moment this method appears sufficient as 
far as the OpenMP specification and current testing can show.
