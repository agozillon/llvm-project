subroutine defaultmap_allocatable_present()
    implicit none
    integer, dimension(:), allocatable :: arr
    integer :: N = 16
    integer :: i

    allocate(arr(N))

!$omp target enter data map(to: arr)

!$omp target defaultmap(present: allocatable)
    do i = 1,N
        arr(i) = 42
    end do
!$omp end target

!$omp target exit data map(from: arr)

    print *, arr
    deallocate(arr)

    return
end subroutine

subroutine defaultmap_scalar_tofrom()
    implicit none
    integer :: scalar_int
    scalar_int = 10

   !$omp target defaultmap(tofrom: scalar)
        scalar_int = 20
   !$omp end target

    print *, scalar_int
    return
end subroutine

subroutine defaultmap_all_default()
    implicit none
    integer, dimension(:), allocatable :: arr
    integer :: aggregate(16)
    integer :: N = 16
    integer :: i, scalar_int

    allocate(arr(N))

    scalar_int = 10
    aggregate = scalar_int

   !$omp target defaultmap(default: all)
        scalar_int = 20
        do i = 1,N
            arr(i) = scalar_int + aggregate(i)
        end do
   !$omp end target

    print *, scalar_int
    print *, arr

    deallocate(arr)
    return
end subroutine

subroutine defaultmap_pointer_to()
    implicit none
    integer, dimension(:), pointer :: arr_ptr(:)
    allocate(arr_ptr(10))
    arr_ptr = 10

    !$omp target defaultmap(to: pointer)
        arr_ptr = 20
    !$omp end target

    print *, arr_ptr
    deallocate(arr_ptr)
    return
end subroutine

subroutine defaultmap_scalar_from()
    implicit none
    integer :: scalar_test
    scalar_test = 10
    !$omp target defaultmap(from: scalar)
        scalar_test = 20
    !$omp end target

    print *, scalar_test
    return
end subroutine

subroutine defaultmap_aggregate_to()
    implicit none
    integer :: aggregate_arr(16)
    integer :: i, scalar_test = 0
    aggregate_arr = 0
    !$omp target map(tofrom: scalar_test) defaultmap(to: aggregate)
        do i = 1,16
            aggregate_arr(i) = i
            scalar_test = scalar_test + aggregate_arr(i)
        enddo
    !$omp end target

    print *, scalar_test
    print *, aggregate_arr
    return
end subroutine

program map_present
    implicit none
    call defaultmap_allocatable_present()
    call defaultmap_scalar_tofrom()
    call defaultmap_all_default()
    call defaultmap_pointer_to()
    call defaultmap_scalar_from()
    call defaultmap_aggregate_to()
end program

! 1) Perhaps a none test, but this would likely be a compiler or runtime error...
! 2) more complicated test with multiple defaultmaps

! none
! all

! $BUILD_DIR/bin/flang --offload-arch=gfx90a -fopenmp  target-defaultmap.f90 -o target-defaultmap.out


! ~/git/cmake-3.30.4/bin/cmake -G"Ninja" \
!   -DFLANG_EXPERIMENTAL_OMP_OFFLOAD_BUILD="host_device" \
!   -DCMAKE_C_COMPILER=/home/agozillo/git/aomp21.0/install/bin/clang \
!   -DCMAKE_CXX_COMPILER=/home/agozillo/git/aomp21.0/install/bin/clang++ \
!      -DCMAKE_CXX_FLAGS='-nogpulib' \
!      -DCMAKE_C_FLAGS='-nogpulib' \
!   -DFLANG_OMP_DEVICE_ARCHITECTURES="gfx90a" \
!   ../runtime/

! ~/git/cmake-3.30.4/bin/cmake -G"Ninja" \
!   -DLLVM_ENABLE_RUNTIMES=flang-rt \
!   -DFLANG_RT_EXPERIMENTAL_OFFLOAD_SUPPORT="OpenMP" \
!   -DCMAKE_C_COMPILER=/home/agozillo/git/aomp21.0/install/bin/clang \
!   -DCMAKE_CXX_COMPILER=/home/agozillo/git/aomp21.0/install/bin/clang++ \
!   -DFLANG_RT_DEVICE_ARCHITECTURES=gfx90a \
!   ../../runtimes/



! ~/git/cmake-3.30.4/bin/cmake -G"Ninja" \
!   -DLLVM_ENABLE_RUNTIMES=flang-rt \
!   -DFLANG_RT_EXPERIMENTAL_OFFLOAD_SUPPORT="OpenMP" \
!   -DCMAKE_C_COMPILER=/home/agozillo/rocm/aomp/llvm/bin/clang \
!   -DCMAKE_CXX_COMPILER=/home/agozillo/rocm/aomp/llvm/bin/clang++ \
!   -DFLANG_RT_DEVICE_ARCHITECTURES=gfx90a \
!   ../../runtimes/
