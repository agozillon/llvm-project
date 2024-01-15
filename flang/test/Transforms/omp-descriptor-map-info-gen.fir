// RUN: fir-opt --omp-descriptor-map-info-gen %s | FileCheck %s

module attributes {omp.is_target_device = false} {
  func.func @test_descriptor_expansion_pass() {
    %0 = fir.alloca !fir.box<!fir.heap<i32>> {bindc_name = "test", uniq_name = "_QFEtest"}
    %1 = fir.zero_bits !fir.heap<i32>
    %2 = fir.embox %1 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
    fir.store %2 to %0 : !fir.ref<!fir.box<!fir.heap<i32>>>
    %3:2 = hlfir.declare %0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFEtest"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)
    %4 = fir.allocmem i32 {fir.must_be_heap = true, uniq_name = "_QFEtest.alloc"}
    %5 = fir.embox %4 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
    fir.store %5 to %3#1 : !fir.ref<!fir.box<!fir.heap<i32>>>
    %6 = omp.map_info var_ptr(%3#1 : !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.box<!fir.heap<i32>>) map_clauses(tofrom) capture(ByRef) -> !fir.ref<!fir.box<!fir.heap<i32>>> {name = "test"}
    omp.target map_entries(%6 -> %arg0 : !fir.ref<!fir.box<!fir.heap<i32>>>) {
    ^bb0(%arg0: !fir.ref<!fir.box<!fir.heap<i32>>>):
      omp.terminator
    }
    return 
  }
}

// CHECK: %[[DECLARE:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFEtest"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)
// CHECK: %[[BASE_ADDR_OFF:.*]] = fir.box_offset %[[DECLARE]]#1 base_addr : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.llvm_ptr<!fir.ref<i32>>
// CHECK: %[[DESC_MEMBER_MAP:.*]] = omp.map_info var_ptr(%[[BASE_ADDR_OFF]] : !fir.llvm_ptr<!fir.ref<i32>>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.llvm_ptr<!fir.ref<i32>> {name = ""}
// CHECK: %[[DESC_PARENT_MAP:.*]] = omp.map_info var_ptr(%[[DECLARE]]#1 : !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.box<!fir.heap<i32>>) map_clauses(tofrom) capture(ByRef) members(%[[DESC_MEMBER_MAP]] : !fir.llvm_ptr<!fir.ref<i32>>) -> !fir.ref<!fir.box<!fir.heap<i32>>> {name = "test"}
// CHECK: omp.target map_entries(%[[DESC_MEMBER_MAP]] -> %{{.*}}, %[[DESC_PARENT_MAP]] -> %{{.*}} : !fir.llvm_ptr<!fir.ref<i32>>, !fir.ref<!fir.box<!fir.heap<i32>>>) {
// CHECK: ^bb0(%{{.*}}: !fir.llvm_ptr<!fir.ref<i32>>, %{{.*}}: !fir.ref<!fir.box<!fir.heap<i32>>>):
