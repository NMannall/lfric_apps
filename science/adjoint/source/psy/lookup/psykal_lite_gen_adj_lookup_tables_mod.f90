!-----------------------------------------------------------------------------
! (C) Crown copyright 2025 Met Office. All rights reserved.
! The file LICENCE, distributed with this code, contains details of the terms
! under which the code may be used.
!-----------------------------------------------------------------------------

!> @brief PSy-lite code required for invokation of the lookup table generation kernels.
!> @detail Halo exchange depths have been manually changed for `lookup_X_field` and
!!         `set_counts_X_field` to halo exchange to the required depth. This is
!!         required because these fields are accessed with a STENCIL, but are
!!         not marked as such. The reason being that PSyclone does not allow
!!         writing to STENCIL fields. We are working around this
!!         by passing a second set of "dummy" fields in READ mode with a STENCIL.
!!         See PSyclone issue #3003.
module psykal_lite_gen_lookup_tables_psy_mod
  USE constants_mod, ONLY: r_tran, r_solver, i_def, r_def
  USE r_tran_field_mod, ONLY: r_tran_field_type, r_tran_field_proxy_type
  USE r_solver_field_mod, ONLY: r_solver_field_type, r_solver_field_proxy_type
  USE integer_field_mod, ONLY: integer_field_type, integer_field_proxy_type
  USE operator_mod, ONLY: operator_type, operator_proxy_type
  IMPLICIT NONE
  CONTAINS

  SUBROUTINE invoke_gen_poly1d_lookup_kernel(tracer, coeff, reconstruction, lookup_poly1d_field, set_counts_poly1d_field, &
&lookup_poly1d_field_dummy, set_counts_poly1d_field_dummy, order, nindices, stencil_extent, loop_halo_depth)
    USE gen_poly1d_lookup_kernel_mod, ONLY: gen_poly1d_lookup_code
    USE mesh_mod, ONLY: mesh_type
    USE stencil_dofmap_mod, ONLY: STENCIL_CROSS
    USE stencil_dofmap_mod, ONLY: stencil_dofmap_type
    INTEGER(KIND=i_def), intent(in) :: order, nindices
    TYPE(r_tran_field_type), intent(in) :: tracer, coeff, reconstruction
    TYPE(integer_field_type), intent(in) :: lookup_poly1d_field, set_counts_poly1d_field, lookup_poly1d_field_dummy, &
&set_counts_poly1d_field_dummy
    INTEGER(KIND=i_def), intent(in) :: stencil_extent
    INTEGER, intent(in) :: loop_halo_depth
    INTEGER(KIND=i_def) cell
    INTEGER(KIND=i_def) loop0_start, loop0_stop
    INTEGER(KIND=i_def) nlayers_tracer
    INTEGER(KIND=i_def), pointer, dimension(:) :: set_counts_poly1d_field_dummy_data
    INTEGER(KIND=i_def), pointer, dimension(:) :: lookup_poly1d_field_dummy_data
    INTEGER(KIND=i_def), pointer, dimension(:) :: set_counts_poly1d_field_data
    INTEGER(KIND=i_def), pointer, dimension(:) :: lookup_poly1d_field_data
    TYPE(integer_field_proxy_type) lookup_poly1d_field_proxy, set_counts_poly1d_field_proxy, lookup_poly1d_field_dummy_proxy, &
&set_counts_poly1d_field_dummy_proxy
    REAL(KIND=r_tran), pointer, dimension(:) :: reconstruction_data
    REAL(KIND=r_tran), pointer, dimension(:) :: coeff_data
    REAL(KIND=r_tran), pointer, dimension(:) :: tracer_data
    TYPE(r_tran_field_proxy_type) tracer_proxy, coeff_proxy, reconstruction_proxy
    INTEGER(KIND=i_def), pointer :: map_adspc1_tracer(:,:), map_adspc2_coeff(:,:), &
&map_adspc3_reconstruction(:,:), map_adspc4_lookup_poly1d_field(:,:), &
&map_adspc5_set_counts_poly1d_field(:,:)
    INTEGER(KIND=i_def) ndf_adspc1_tracer, undf_adspc1_tracer, ndf_adspc2_coeff, undf_adspc2_coeff, ndf_adspc3_reconstruction, &
&undf_adspc3_reconstruction, ndf_adspc4_lookup_poly1d_field, undf_adspc4_lookup_poly1d_field, ndf_adspc5_set_counts_poly1d_field, &
&undf_adspc5_set_counts_poly1d_field
    INTEGER(KIND=i_def) max_halo_depth_mesh
    TYPE(mesh_type), pointer :: mesh
    INTEGER(KIND=i_def), pointer :: set_counts_poly1d_field_dummy_stencil_size(:)
    INTEGER(KIND=i_def), pointer :: set_counts_poly1d_field_dummy_stencil_dofmap(:,:,:)
    TYPE(stencil_dofmap_type), pointer :: set_counts_poly1d_field_dummy_stencil_map
    INTEGER(KIND=i_def), pointer :: lookup_poly1d_field_dummy_stencil_size(:)
    INTEGER(KIND=i_def), pointer :: lookup_poly1d_field_dummy_stencil_dofmap(:,:,:)
    TYPE(stencil_dofmap_type), pointer :: lookup_poly1d_field_dummy_stencil_map

    nullify( set_counts_poly1d_field_dummy_data, lookup_poly1d_field_dummy_data, set_counts_poly1d_field_data, &
             lookup_poly1d_field_data, reconstruction_data, coeff_data, tracer_data, &
             map_adspc1_tracer, map_adspc2_coeff, map_adspc4_lookup_poly1d_field, &
             mesh, set_counts_poly1d_field_dummy_stencil_size, set_counts_poly1d_field_dummy_stencil_dofmap, &
             set_counts_poly1d_field_dummy_stencil_map, lookup_poly1d_field_dummy_stencil_size, &
             lookup_poly1d_field_dummy_stencil_dofmap, lookup_poly1d_field_dummy_stencil_map)

    !
    ! Initialise field and/or operator proxies
    !
    tracer_proxy = tracer%get_proxy()
    tracer_data => tracer_proxy%data
    coeff_proxy = coeff%get_proxy()
    coeff_data => coeff_proxy%data
    reconstruction_proxy = reconstruction%get_proxy()
    reconstruction_data => reconstruction_proxy%data
    lookup_poly1d_field_proxy = lookup_poly1d_field%get_proxy()
    lookup_poly1d_field_data => lookup_poly1d_field_proxy%data
    set_counts_poly1d_field_proxy = set_counts_poly1d_field%get_proxy()
    set_counts_poly1d_field_data => set_counts_poly1d_field_proxy%data
    lookup_poly1d_field_dummy_proxy = lookup_poly1d_field_dummy%get_proxy()
    lookup_poly1d_field_dummy_data => lookup_poly1d_field_dummy_proxy%data
    set_counts_poly1d_field_dummy_proxy = set_counts_poly1d_field_dummy%get_proxy()
    set_counts_poly1d_field_dummy_data => set_counts_poly1d_field_dummy_proxy%data
    !
    ! Initialise number of layers
    !
    nlayers_tracer = tracer_proxy%vspace%get_nlayers()
    !
    ! Create a mesh object
    !
    mesh => tracer_proxy%vspace%get_mesh()
    max_halo_depth_mesh = mesh%get_halo_depth()
    !
    ! Initialise stencil dofmaps
    !
    lookup_poly1d_field_dummy_stencil_map => &
&lookup_poly1d_field_dummy_proxy%vspace%get_stencil_dofmap(STENCIL_CROSS,stencil_extent)
    lookup_poly1d_field_dummy_stencil_dofmap => lookup_poly1d_field_dummy_stencil_map%get_whole_dofmap()
    lookup_poly1d_field_dummy_stencil_size => lookup_poly1d_field_dummy_stencil_map%get_stencil_sizes()
    set_counts_poly1d_field_dummy_stencil_map => &
&set_counts_poly1d_field_dummy_proxy%vspace%get_stencil_dofmap(STENCIL_CROSS,stencil_extent)
    set_counts_poly1d_field_dummy_stencil_dofmap => set_counts_poly1d_field_dummy_stencil_map%get_whole_dofmap()
    set_counts_poly1d_field_dummy_stencil_size => set_counts_poly1d_field_dummy_stencil_map%get_stencil_sizes()
    !
    ! Look-up dofmaps for each function space
    !
    map_adspc1_tracer => tracer_proxy%vspace%get_whole_dofmap()
    map_adspc2_coeff => coeff_proxy%vspace%get_whole_dofmap()
    map_adspc3_reconstruction => reconstruction_proxy%vspace%get_whole_dofmap()
    map_adspc4_lookup_poly1d_field => lookup_poly1d_field_proxy%vspace%get_whole_dofmap()
    map_adspc5_set_counts_poly1d_field => set_counts_poly1d_field_proxy%vspace%get_whole_dofmap()
    !
    ! Initialise number of DoFs for adspc1_tracer
    !
    ndf_adspc1_tracer = tracer_proxy%vspace%get_ndf()
    undf_adspc1_tracer = tracer_proxy%vspace%get_undf()
    !
    ! Initialise number of DoFs for adspc2_coeff
    !
    ndf_adspc2_coeff = coeff_proxy%vspace%get_ndf()
    undf_adspc2_coeff = coeff_proxy%vspace%get_undf()
    !
    ! Initialise number of DoFs for adspc3_reconstruction
    !
    ndf_adspc3_reconstruction = reconstruction_proxy%vspace%get_ndf()
    undf_adspc3_reconstruction = reconstruction_proxy%vspace%get_undf()
    !
    ! Initialise number of DoFs for adspc4_lookup_poly1d_field
    !
    ndf_adspc4_lookup_poly1d_field = lookup_poly1d_field_proxy%vspace%get_ndf()
    undf_adspc4_lookup_poly1d_field = lookup_poly1d_field_proxy%vspace%get_undf()
    !
    ! Initialise number of DoFs for adspc5_set_counts_poly1d_field
    !
    ndf_adspc5_set_counts_poly1d_field = set_counts_poly1d_field_proxy%vspace%get_ndf()
    undf_adspc5_set_counts_poly1d_field = set_counts_poly1d_field_proxy%vspace%get_undf()
    !
    ! Set-up all of the loop bounds
    !
    loop0_start = 1
    loop0_stop = mesh%get_last_halo_cell(loop_halo_depth)
    !
    ! Call kernels and communication routines
    !
    IF (tracer_proxy%is_dirty(depth=loop_halo_depth)) THEN
      CALL tracer_proxy%halo_exchange(depth=loop_halo_depth)
    END IF
    IF (coeff_proxy%is_dirty(depth=loop_halo_depth)) THEN
      CALL coeff_proxy%halo_exchange(depth=loop_halo_depth)
    END IF
    IF (reconstruction_proxy%is_dirty(depth=loop_halo_depth)) THEN
      CALL reconstruction_proxy%halo_exchange(depth=loop_halo_depth)
    END IF
    IF (lookup_poly1d_field_proxy%is_dirty(depth=loop_halo_depth + stencil_extent)) THEN
      CALL lookup_poly1d_field_proxy%halo_exchange(depth=loop_halo_depth + stencil_extent)
    END IF
    IF (set_counts_poly1d_field_proxy%is_dirty(depth=loop_halo_depth + stencil_extent)) THEN
      CALL set_counts_poly1d_field_proxy%halo_exchange(depth=loop_halo_depth + stencil_extent)
    END IF
    DO cell = loop0_start, loop0_stop, 1
      CALL gen_poly1d_lookup_code(nlayers_tracer, tracer_data, coeff_data, reconstruction_data, lookup_poly1d_field_data, &
&set_counts_poly1d_field_data, lookup_poly1d_field_dummy_data, lookup_poly1d_field_dummy_stencil_size(cell), &
&lookup_poly1d_field_dummy_stencil_dofmap(:,:,cell), set_counts_poly1d_field_dummy_data, &
&set_counts_poly1d_field_dummy_stencil_size(cell), set_counts_poly1d_field_dummy_stencil_dofmap(:,:,cell), order, nindices, &
&ndf_adspc1_tracer, undf_adspc1_tracer, map_adspc1_tracer(:,cell), ndf_adspc2_coeff, undf_adspc2_coeff, map_adspc2_coeff(:,cell), &
&ndf_adspc3_reconstruction, undf_adspc3_reconstruction, map_adspc3_reconstruction(:,cell), ndf_adspc4_lookup_poly1d_field, &
&undf_adspc4_lookup_poly1d_field, map_adspc4_lookup_poly1d_field(:,cell), ndf_adspc5_set_counts_poly1d_field, &
&undf_adspc5_set_counts_poly1d_field, map_adspc5_set_counts_poly1d_field(:,cell))
    END DO
    !
    ! Set halos dirty/clean for fields modified in the above loop
    !
    CALL lookup_poly1d_field_proxy%set_dirty()
    CALL lookup_poly1d_field_proxy%set_clean(loop_halo_depth)
    CALL set_counts_poly1d_field_proxy%set_dirty()
    CALL set_counts_poly1d_field_proxy%set_clean(loop_halo_depth)
    !
    !
  END SUBROUTINE invoke_gen_poly1d_lookup_kernel

  SUBROUTINE invoke_gen_poly2d_lookup_kernel(tracer, coeff, reconstruction, lookup_poly2d_field, set_counts_poly2d_field, &
&lookup_poly2d_field_dummy, set_counts_poly2d_field_dummy, stencil_size, nindices, stencil_extent, loop_halo_depth)
    USE gen_poly2d_lookup_kernel_mod, ONLY: gen_poly2d_lookup_code
    USE mesh_mod, ONLY: mesh_type
    USE stencil_dofmap_mod, ONLY: STENCIL_REGION
    USE stencil_dofmap_mod, ONLY: stencil_dofmap_type
    INTEGER(KIND=i_def), intent(in) :: stencil_size, nindices
    TYPE(r_tran_field_type), intent(in) :: tracer, coeff, reconstruction
    TYPE(integer_field_type), intent(in) :: lookup_poly2d_field, set_counts_poly2d_field, lookup_poly2d_field_dummy, &
&set_counts_poly2d_field_dummy
    INTEGER(KIND=i_def), intent(in) :: stencil_extent
    INTEGER, intent(in) :: loop_halo_depth
    INTEGER(KIND=i_def) cell
    INTEGER(KIND=i_def) loop0_start, loop0_stop
    INTEGER(KIND=i_def) nlayers_tracer
    INTEGER(KIND=i_def), pointer, dimension(:) :: set_counts_poly2d_field_dummy_data
    INTEGER(KIND=i_def), pointer, dimension(:) :: lookup_poly2d_field_dummy_data
    INTEGER(KIND=i_def), pointer, dimension(:) :: set_counts_poly2d_field_data
    INTEGER(KIND=i_def), pointer, dimension(:) :: lookup_poly2d_field_data
    TYPE(integer_field_proxy_type) lookup_poly2d_field_proxy, set_counts_poly2d_field_proxy, lookup_poly2d_field_dummy_proxy, &
&set_counts_poly2d_field_dummy_proxy
    REAL(KIND=r_tran), pointer, dimension(:) :: reconstruction_data
    REAL(KIND=r_tran), pointer, dimension(:) :: coeff_data
    REAL(KIND=r_tran), pointer, dimension(:) :: tracer_data
    TYPE(r_tran_field_proxy_type) tracer_proxy, coeff_proxy, reconstruction_proxy
    INTEGER(KIND=i_def), pointer :: map_adspc1_tracer(:,:), map_adspc2_coeff(:,:), &
&map_adspc3_reconstruction(:,:), map_adspc4_lookup_poly2d_field(:,:), &
&map_adspc5_set_counts_poly2d_field(:,:)
    INTEGER(KIND=i_def) ndf_adspc1_tracer, undf_adspc1_tracer, ndf_adspc2_coeff, undf_adspc2_coeff, ndf_adspc3_reconstruction, &
&undf_adspc3_reconstruction, ndf_adspc4_lookup_poly2d_field, undf_adspc4_lookup_poly2d_field, ndf_adspc5_set_counts_poly2d_field, &
&undf_adspc5_set_counts_poly2d_field
    INTEGER(KIND=i_def) max_halo_depth_mesh
    TYPE(mesh_type), pointer :: mesh
    INTEGER(KIND=i_def), pointer :: set_counts_poly2d_field_dummy_stencil_size(:)
    INTEGER(KIND=i_def), pointer :: set_counts_poly2d_field_dummy_stencil_dofmap(:,:,:)
    TYPE(stencil_dofmap_type), pointer :: set_counts_poly2d_field_dummy_stencil_map
    INTEGER(KIND=i_def), pointer :: lookup_poly2d_field_dummy_stencil_size(:)
    INTEGER(KIND=i_def), pointer :: lookup_poly2d_field_dummy_stencil_dofmap(:,:,:)
    TYPE(stencil_dofmap_type), pointer :: lookup_poly2d_field_dummy_stencil_map

    nullify( set_counts_poly2d_field_dummy_data, lookup_poly2d_field_dummy_data, set_counts_poly2d_field_data, &
             lookup_poly2d_field_data, reconstruction_data, coeff_data, tracer_data, &
             map_adspc1_tracer, map_adspc2_coeff, map_adspc4_lookup_poly2d_field, &
             mesh, set_counts_poly2d_field_dummy_stencil_size, set_counts_poly2d_field_dummy_stencil_dofmap, &
             set_counts_poly2d_field_dummy_stencil_map, lookup_poly2d_field_dummy_stencil_size, &
             lookup_poly2d_field_dummy_stencil_dofmap, lookup_poly2d_field_dummy_stencil_map)

    !
    ! Initialise field and/or operator proxies
    !
    tracer_proxy = tracer%get_proxy()
    tracer_data => tracer_proxy%data
    coeff_proxy = coeff%get_proxy()
    coeff_data => coeff_proxy%data
    reconstruction_proxy = reconstruction%get_proxy()
    reconstruction_data => reconstruction_proxy%data
    lookup_poly2d_field_proxy = lookup_poly2d_field%get_proxy()
    lookup_poly2d_field_data => lookup_poly2d_field_proxy%data
    set_counts_poly2d_field_proxy = set_counts_poly2d_field%get_proxy()
    set_counts_poly2d_field_data => set_counts_poly2d_field_proxy%data
    lookup_poly2d_field_dummy_proxy = lookup_poly2d_field_dummy%get_proxy()
    lookup_poly2d_field_dummy_data => lookup_poly2d_field_dummy_proxy%data
    set_counts_poly2d_field_dummy_proxy = set_counts_poly2d_field_dummy%get_proxy()
    set_counts_poly2d_field_dummy_data => set_counts_poly2d_field_dummy_proxy%data
    !
    ! Initialise number of layers
    !
    nlayers_tracer = tracer_proxy%vspace%get_nlayers()
    !
    ! Create a mesh object
    !
    mesh => tracer_proxy%vspace%get_mesh()
    max_halo_depth_mesh = mesh%get_halo_depth()
    !
    ! Initialise stencil dofmaps
    !
    lookup_poly2d_field_dummy_stencil_map => &
&lookup_poly2d_field_dummy_proxy%vspace%get_stencil_dofmap(STENCIL_REGION,stencil_extent)
    lookup_poly2d_field_dummy_stencil_dofmap => lookup_poly2d_field_dummy_stencil_map%get_whole_dofmap()
    lookup_poly2d_field_dummy_stencil_size => lookup_poly2d_field_dummy_stencil_map%get_stencil_sizes()
    set_counts_poly2d_field_dummy_stencil_map => &
&set_counts_poly2d_field_dummy_proxy%vspace%get_stencil_dofmap(STENCIL_REGION,stencil_extent)
    set_counts_poly2d_field_dummy_stencil_dofmap => set_counts_poly2d_field_dummy_stencil_map%get_whole_dofmap()
    set_counts_poly2d_field_dummy_stencil_size => set_counts_poly2d_field_dummy_stencil_map%get_stencil_sizes()
    !
    ! Look-up dofmaps for each function space
    !
    map_adspc1_tracer => tracer_proxy%vspace%get_whole_dofmap()
    map_adspc2_coeff => coeff_proxy%vspace%get_whole_dofmap()
    map_adspc3_reconstruction => reconstruction_proxy%vspace%get_whole_dofmap()
    map_adspc4_lookup_poly2d_field => lookup_poly2d_field_proxy%vspace%get_whole_dofmap()
    map_adspc5_set_counts_poly2d_field => set_counts_poly2d_field_proxy%vspace%get_whole_dofmap()
    !
    ! Initialise number of DoFs for adspc1_tracer
    !
    ndf_adspc1_tracer = tracer_proxy%vspace%get_ndf()
    undf_adspc1_tracer = tracer_proxy%vspace%get_undf()
    !
    ! Initialise number of DoFs for adspc2_coeff
    !
    ndf_adspc2_coeff = coeff_proxy%vspace%get_ndf()
    undf_adspc2_coeff = coeff_proxy%vspace%get_undf()
    !
    ! Initialise number of DoFs for adspc3_reconstruction
    !
    ndf_adspc3_reconstruction = reconstruction_proxy%vspace%get_ndf()
    undf_adspc3_reconstruction = reconstruction_proxy%vspace%get_undf()
    !
    ! Initialise number of DoFs for adspc4_lookup_poly2d_field
    !
    ndf_adspc4_lookup_poly2d_field = lookup_poly2d_field_proxy%vspace%get_ndf()
    undf_adspc4_lookup_poly2d_field = lookup_poly2d_field_proxy%vspace%get_undf()
    !
    ! Initialise number of DoFs for adspc5_set_counts_poly2d_field
    !
    ndf_adspc5_set_counts_poly2d_field = set_counts_poly2d_field_proxy%vspace%get_ndf()
    undf_adspc5_set_counts_poly2d_field = set_counts_poly2d_field_proxy%vspace%get_undf()
    !
    ! Set-up all of the loop bounds
    !
    loop0_start = 1
    loop0_stop = mesh%get_last_halo_cell(loop_halo_depth)
    !
    ! Call kernels and communication routines
    !
    IF (tracer_proxy%is_dirty(depth=loop_halo_depth)) THEN
      CALL tracer_proxy%halo_exchange(depth=loop_halo_depth)
    END IF
    IF (coeff_proxy%is_dirty(depth=loop_halo_depth)) THEN
      CALL coeff_proxy%halo_exchange(depth=loop_halo_depth)
    END IF
    IF (reconstruction_proxy%is_dirty(depth=loop_halo_depth)) THEN
      CALL reconstruction_proxy%halo_exchange(depth=loop_halo_depth)
    END IF
    IF (lookup_poly2d_field_proxy%is_dirty(depth=loop_halo_depth + stencil_extent)) THEN
      CALL lookup_poly2d_field_proxy%halo_exchange(depth=loop_halo_depth + stencil_extent)
    END IF
    IF (set_counts_poly2d_field_proxy%is_dirty(depth=loop_halo_depth + stencil_extent)) THEN
      CALL set_counts_poly2d_field_proxy%halo_exchange(depth=loop_halo_depth + stencil_extent)
    END IF
    DO cell = loop0_start, loop0_stop, 1
      CALL gen_poly2d_lookup_code(nlayers_tracer, tracer_data, coeff_data, reconstruction_data, lookup_poly2d_field_data, &
&set_counts_poly2d_field_data, lookup_poly2d_field_dummy_data, lookup_poly2d_field_dummy_stencil_size(cell), &
&lookup_poly2d_field_dummy_stencil_dofmap(:,:,cell), set_counts_poly2d_field_dummy_data, &
&set_counts_poly2d_field_dummy_stencil_size(cell), set_counts_poly2d_field_dummy_stencil_dofmap(:,:,cell), stencil_size, nindices, &
&ndf_adspc1_tracer, undf_adspc1_tracer, map_adspc1_tracer(:,cell), ndf_adspc2_coeff, undf_adspc2_coeff, map_adspc2_coeff(:,cell), &
&ndf_adspc3_reconstruction, undf_adspc3_reconstruction, map_adspc3_reconstruction(:,cell), ndf_adspc4_lookup_poly2d_field, &
&undf_adspc4_lookup_poly2d_field, map_adspc4_lookup_poly2d_field(:,cell), ndf_adspc5_set_counts_poly2d_field, &
&undf_adspc5_set_counts_poly2d_field, map_adspc5_set_counts_poly2d_field(:,cell))
    END DO
    !
    ! Set halos dirty/clean for fields modified in the above loop
    !
    CALL lookup_poly2d_field_proxy%set_dirty()
    CALL lookup_poly2d_field_proxy%set_clean(loop_halo_depth)
    CALL set_counts_poly2d_field_proxy%set_dirty()
    CALL set_counts_poly2d_field_proxy%set_clean(loop_halo_depth)
    !
    !
  END SUBROUTINE invoke_gen_poly2d_lookup_kernel

  SUBROUTINE invoke_gen_poly_adv_upd_lookup_kernel(advective, reconstruction_big_halo, wind_big_halo, lookup_field, &
&set_counts_field, lookup_field_dummy, set_counts_field_dummy, nsets_max, nindices, stencil_extent, loop_halo_depth)
    USE gen_poly_adv_upd_lookup_kernel_mod, ONLY: gen_poly_adv_upd_lookup_code
    USE mesh_mod, ONLY: mesh_type
    USE stencil_2D_dofmap_mod, ONLY: stencil_2D_dofmap_type, STENCIL_2D_CROSS
    INTEGER(KIND=i_def), intent(in) :: nsets_max, nindices
    TYPE(r_tran_field_type), intent(in) :: advective, reconstruction_big_halo, wind_big_halo
    TYPE(integer_field_type), intent(in) :: lookup_field, set_counts_field, lookup_field_dummy, set_counts_field_dummy
    INTEGER(KIND=i_def), intent(in) :: stencil_extent
    INTEGER, intent(in) :: loop_halo_depth
    INTEGER(KIND=i_def) cell
    INTEGER(KIND=i_def) loop0_start, loop0_stop
    INTEGER(KIND=i_def) nlayers_advective
    INTEGER(KIND=i_def), pointer, dimension(:) :: set_counts_field_dummy_data
    INTEGER(KIND=i_def), pointer, dimension(:) :: lookup_field_dummy_data
    INTEGER(KIND=i_def), pointer, dimension(:) :: set_counts_field_data
    INTEGER(KIND=i_def), pointer, dimension(:) :: lookup_field_data
    TYPE(integer_field_proxy_type) lookup_field_proxy, set_counts_field_proxy, lookup_field_dummy_proxy, &
&set_counts_field_dummy_proxy
    REAL(KIND=r_tran), pointer, dimension(:) :: wind_big_halo_data
    REAL(KIND=r_tran), pointer, dimension(:) :: reconstruction_big_halo_data
    REAL(KIND=r_tran), pointer, dimension(:) :: advective_data
    TYPE(r_tran_field_proxy_type) advective_proxy, reconstruction_big_halo_proxy, wind_big_halo_proxy
    INTEGER(KIND=i_def), pointer :: map_adspc1_reconstruction_big_halo(:,:), map_adspc2_lookup_field(:,:), &
&map_adspc3_set_counts_field(:,:), map_w2(:,:), map_wtheta(:,:)
    INTEGER(KIND=i_def) ndf_wtheta, undf_wtheta, ndf_adspc1_reconstruction_big_halo, undf_adspc1_reconstruction_big_halo, &
&ndf_w2, undf_w2, ndf_adspc2_lookup_field, undf_adspc2_lookup_field, ndf_adspc3_set_counts_field, undf_adspc3_set_counts_field
    INTEGER(KIND=i_def) max_halo_depth_mesh
    TYPE(mesh_type), pointer :: mesh
    INTEGER(KIND=i_def) set_counts_field_dummy_max_branch_length
    INTEGER(KIND=i_def), pointer :: set_counts_field_dummy_stencil_size(:,:)
    INTEGER(KIND=i_def), pointer :: set_counts_field_dummy_stencil_dofmap(:,:,:,:)
    TYPE(stencil_2D_dofmap_type), pointer :: set_counts_field_dummy_stencil_map
    INTEGER(KIND=i_def) lookup_field_dummy_max_branch_length
    INTEGER(KIND=i_def), pointer :: lookup_field_dummy_stencil_size(:,:)
    INTEGER(KIND=i_def), pointer :: lookup_field_dummy_stencil_dofmap(:,:,:,:)
    TYPE(stencil_2D_dofmap_type), pointer :: lookup_field_dummy_stencil_map
    INTEGER(KIND=i_def) wind_big_halo_max_branch_length
    INTEGER(KIND=i_def), pointer :: wind_big_halo_stencil_size(:,:)
    INTEGER(KIND=i_def), pointer :: wind_big_halo_stencil_dofmap(:,:,:,:)
    TYPE(stencil_2D_dofmap_type), pointer :: wind_big_halo_stencil_map
    INTEGER(KIND=i_def) reconstruction_big_halo_max_branch_length
    INTEGER(KIND=i_def), pointer :: reconstruction_big_halo_stencil_size(:,:)
    INTEGER(KIND=i_def), pointer :: reconstruction_big_halo_stencil_dofmap(:,:,:,:)
    TYPE(stencil_2D_dofmap_type), pointer :: reconstruction_big_halo_stencil_map

    nullify( set_counts_field_dummy_data, lookup_field_dummy_data, set_counts_field_data, &
             lookup_field_data, wind_big_halo_data, reconstruction_big_halo_data, advective_data, &
             map_adspc1_reconstruction_big_halo, map_adspc2_lookup_field, map_w2, &
             map_wtheta, mesh, set_counts_field_dummy_stencil_size, set_counts_field_dummy_stencil_dofmap, &
             set_counts_field_dummy_stencil_map, lookup_field_dummy_stencil_size, &
             lookup_field_dummy_stencil_dofmap, lookup_field_dummy_stencil_map, wind_big_halo_stencil_size, &
             wind_big_halo_stencil_dofmap, wind_big_halo_stencil_map, reconstruction_big_halo_stencil_size, &
             reconstruction_big_halo_stencil_dofmap, reconstruction_big_halo_stencil_map )

    !
    ! Initialise field and/or operator proxies
    !
    advective_proxy = advective%get_proxy()
    advective_data => advective_proxy%data
    reconstruction_big_halo_proxy = reconstruction_big_halo%get_proxy()
    reconstruction_big_halo_data => reconstruction_big_halo_proxy%data
    wind_big_halo_proxy = wind_big_halo%get_proxy()
    wind_big_halo_data => wind_big_halo_proxy%data
    lookup_field_proxy = lookup_field%get_proxy()
    lookup_field_data => lookup_field_proxy%data
    set_counts_field_proxy = set_counts_field%get_proxy()
    set_counts_field_data => set_counts_field_proxy%data
    lookup_field_dummy_proxy = lookup_field_dummy%get_proxy()
    lookup_field_dummy_data => lookup_field_dummy_proxy%data
    set_counts_field_dummy_proxy = set_counts_field_dummy%get_proxy()
    set_counts_field_dummy_data => set_counts_field_dummy_proxy%data
    !
    ! Initialise number of layers
    !
    nlayers_advective = advective_proxy%vspace%get_nlayers()
    !
    ! Create a mesh object
    !
    mesh => advective_proxy%vspace%get_mesh()
    max_halo_depth_mesh = mesh%get_halo_depth()
    !
    ! Initialise stencil dofmaps
    !
    reconstruction_big_halo_stencil_map => &
&reconstruction_big_halo_proxy%vspace%get_stencil_2D_dofmap(STENCIL_2D_CROSS,stencil_extent)
    reconstruction_big_halo_max_branch_length = stencil_extent + 1_i_def
    reconstruction_big_halo_stencil_dofmap => reconstruction_big_halo_stencil_map%get_whole_dofmap()
    reconstruction_big_halo_stencil_size => reconstruction_big_halo_stencil_map%get_stencil_sizes()
    wind_big_halo_stencil_map => wind_big_halo_proxy%vspace%get_stencil_2D_dofmap(STENCIL_2D_CROSS,stencil_extent)
    wind_big_halo_max_branch_length = stencil_extent + 1_i_def
    wind_big_halo_stencil_dofmap => wind_big_halo_stencil_map%get_whole_dofmap()
    wind_big_halo_stencil_size => wind_big_halo_stencil_map%get_stencil_sizes()
    lookup_field_dummy_stencil_map => lookup_field_dummy_proxy%vspace%get_stencil_2D_dofmap(STENCIL_2D_CROSS,stencil_extent)
    lookup_field_dummy_max_branch_length = stencil_extent + 1_i_def
    lookup_field_dummy_stencil_dofmap => lookup_field_dummy_stencil_map%get_whole_dofmap()
    lookup_field_dummy_stencil_size => lookup_field_dummy_stencil_map%get_stencil_sizes()
    set_counts_field_dummy_stencil_map => &
&set_counts_field_dummy_proxy%vspace%get_stencil_2D_dofmap(STENCIL_2D_CROSS,stencil_extent)
    set_counts_field_dummy_max_branch_length = stencil_extent + 1_i_def
    set_counts_field_dummy_stencil_dofmap => set_counts_field_dummy_stencil_map%get_whole_dofmap()
    set_counts_field_dummy_stencil_size => set_counts_field_dummy_stencil_map%get_stencil_sizes()
    !
    ! Look-up dofmaps for each function space
    !
    map_wtheta => advective_proxy%vspace%get_whole_dofmap()
    map_adspc1_reconstruction_big_halo => reconstruction_big_halo_proxy%vspace%get_whole_dofmap()
    map_w2 => wind_big_halo_proxy%vspace%get_whole_dofmap()
    map_adspc2_lookup_field => lookup_field_proxy%vspace%get_whole_dofmap()
    map_adspc3_set_counts_field => set_counts_field_proxy%vspace%get_whole_dofmap()
    !
    ! Initialise number of DoFs for wtheta
    !
    ndf_wtheta = advective_proxy%vspace%get_ndf()
    undf_wtheta = advective_proxy%vspace%get_undf()
    !
    ! Initialise number of DoFs for adspc1_reconstruction_big_halo
    !
    ndf_adspc1_reconstruction_big_halo = reconstruction_big_halo_proxy%vspace%get_ndf()
    undf_adspc1_reconstruction_big_halo = reconstruction_big_halo_proxy%vspace%get_undf()
    !
    ! Initialise number of DoFs for w2
    !
    ndf_w2 = wind_big_halo_proxy%vspace%get_ndf()
    undf_w2 = wind_big_halo_proxy%vspace%get_undf()
    !
    ! Initialise number of DoFs for adspc2_lookup_field
    !
    ndf_adspc2_lookup_field = lookup_field_proxy%vspace%get_ndf()
    undf_adspc2_lookup_field = lookup_field_proxy%vspace%get_undf()
    !
    ! Initialise number of DoFs for adspc3_set_counts_field
    !
    ndf_adspc3_set_counts_field = set_counts_field_proxy%vspace%get_ndf()
    undf_adspc3_set_counts_field = set_counts_field_proxy%vspace%get_undf()
    !
    ! Set-up all of the loop bounds
    !
    loop0_start = 1
    loop0_stop = mesh%get_last_halo_cell(loop_halo_depth)
    !
    ! Call kernels and communication routines
    !
    IF (advective_proxy%is_dirty(depth=loop_halo_depth)) THEN
      CALL advective_proxy%halo_exchange(depth=loop_halo_depth)
    END IF
    IF (reconstruction_big_halo_proxy%is_dirty(depth=loop_halo_depth + stencil_extent)) THEN
      CALL reconstruction_big_halo_proxy%halo_exchange(depth=loop_halo_depth + stencil_extent)
    END IF
    IF (wind_big_halo_proxy%is_dirty(depth=loop_halo_depth + stencil_extent)) THEN
      CALL wind_big_halo_proxy%halo_exchange(depth=loop_halo_depth + stencil_extent)
    END IF
    IF (lookup_field_proxy%is_dirty(depth=loop_halo_depth + stencil_extent)) THEN
      CALL lookup_field_proxy%halo_exchange(depth=loop_halo_depth + stencil_extent)
    END IF
    IF (set_counts_field_proxy%is_dirty(depth=loop_halo_depth + stencil_extent)) THEN
      CALL set_counts_field_proxy%halo_exchange(depth=loop_halo_depth + stencil_extent)
    END IF
    DO cell = loop0_start, loop0_stop, 1
      CALL gen_poly_adv_upd_lookup_code(nlayers_advective, advective_data, reconstruction_big_halo_data, &
&reconstruction_big_halo_stencil_size(:,cell), reconstruction_big_halo_max_branch_length, &
&reconstruction_big_halo_stencil_dofmap(:,:,:,cell), wind_big_halo_data, wind_big_halo_stencil_size(:,cell), &
&wind_big_halo_max_branch_length, wind_big_halo_stencil_dofmap(:,:,:,cell), lookup_field_data, set_counts_field_data, &
&lookup_field_dummy_data, lookup_field_dummy_stencil_size(:,cell), lookup_field_dummy_max_branch_length, &
&lookup_field_dummy_stencil_dofmap(:,:,:,cell), set_counts_field_dummy_data, set_counts_field_dummy_stencil_size(:,cell), &
&set_counts_field_dummy_max_branch_length, set_counts_field_dummy_stencil_dofmap(:,:,:,cell), nsets_max, nindices, ndf_wtheta, &
&undf_wtheta, map_wtheta(:,cell), ndf_adspc1_reconstruction_big_halo, undf_adspc1_reconstruction_big_halo, &
&map_adspc1_reconstruction_big_halo(:,cell), ndf_w2, undf_w2, map_w2(:,cell), ndf_adspc2_lookup_field, undf_adspc2_lookup_field, &
&map_adspc2_lookup_field(:,cell), ndf_adspc3_set_counts_field, undf_adspc3_set_counts_field, map_adspc3_set_counts_field(:,cell))
    END DO
    !
    ! Set halos dirty/clean for fields modified in the above loop
    !
    CALL lookup_field_proxy%set_dirty()
    CALL lookup_field_proxy%set_clean(loop_halo_depth)
    CALL set_counts_field_proxy%set_dirty()
    CALL set_counts_field_proxy%set_clean(loop_halo_depth)
    !
    !
  END SUBROUTINE invoke_gen_poly_adv_upd_lookup_kernel

  !> @brief Additional edit required due to PSyclone issue #2934. Use of integer fields with LMA operators.
  SUBROUTINE invoke_gen_w3h_adv_upd_lookup_kernel(advective_increment, wind, m3_inv, lookup_field, set_counts_field, lookup_field_dummy, &
&set_counts_field_dummy, nsets_max, nindices, stencil_extent, loop_halo_depth)
    USE gen_w3h_adv_upd_lookup_kernel_mod, ONLY: gen_w3h_adv_upd_lookup_code
    USE mesh_mod, ONLY: mesh_type
    USE stencil_2D_dofmap_mod, ONLY: stencil_2D_dofmap_type, STENCIL_2D_CROSS
    INTEGER(KIND=i_def), intent(in) :: nsets_max, nindices
    TYPE(r_tran_field_type), intent(in) :: advective_increment, wind
    TYPE(operator_type), intent(in) :: m3_inv
    TYPE(integer_field_type), intent(in) :: lookup_field, set_counts_field, lookup_field_dummy, set_counts_field_dummy
    INTEGER(KIND=i_def), intent(in) :: stencil_extent
    INTEGER, intent(in) :: loop_halo_depth
    INTEGER(KIND=i_def) cell
    INTEGER(KIND=i_def) loop0_start, loop0_stop
    INTEGER(KIND=i_def) nlayers_advective_increment
    REAL(KIND=r_def), pointer, dimension(:,:,:) :: m3_inv_local_stencil
    TYPE(operator_proxy_type) m3_inv_proxy
    INTEGER(KIND=i_def), pointer, dimension(:) :: set_counts_field_dummy_data
    INTEGER(KIND=i_def), pointer, dimension(:) :: lookup_field_dummy_data
    INTEGER(KIND=i_def), pointer, dimension(:) :: set_counts_field_data
    INTEGER(KIND=i_def), pointer, dimension(:) :: lookup_field_data
    TYPE(integer_field_proxy_type) lookup_field_proxy, set_counts_field_proxy, lookup_field_dummy_proxy, &
&set_counts_field_dummy_proxy
    REAL(KIND=r_tran), pointer, dimension(:) :: wind_data
    REAL(KIND=r_tran), pointer, dimension(:) :: advective_increment_data
    TYPE(r_tran_field_proxy_type) advective_increment_proxy, wind_proxy
    INTEGER(KIND=i_def), pointer :: map_adspc1_lookup_field(:,:), map_adspc2_set_counts_field(:,:), &
&map_any_w2(:,:), map_w3(:,:)
    INTEGER(KIND=i_def) ndf_w3, undf_w3, ndf_any_w2, undf_any_w2, ndf_adspc1_lookup_field, undf_adspc1_lookup_field, &
&ndf_adspc2_set_counts_field, undf_adspc2_set_counts_field
    INTEGER(KIND=i_def) max_halo_depth_mesh
    TYPE(mesh_type), pointer :: mesh
    INTEGER(KIND=i_def) set_counts_field_dummy_max_branch_length
    INTEGER(KIND=i_def), pointer :: set_counts_field_dummy_stencil_size(:,:)
    INTEGER(KIND=i_def), pointer :: set_counts_field_dummy_stencil_dofmap(:,:,:,:)
    TYPE(stencil_2D_dofmap_type), pointer :: set_counts_field_dummy_stencil_map
    INTEGER(KIND=i_def) lookup_field_dummy_max_branch_length
    INTEGER(KIND=i_def), pointer :: lookup_field_dummy_stencil_size(:,:)
    INTEGER(KIND=i_def), pointer :: lookup_field_dummy_stencil_dofmap(:,:,:,:)
    TYPE(stencil_2D_dofmap_type), pointer :: lookup_field_dummy_stencil_map
    INTEGER(KIND=i_def) wind_max_branch_length
    INTEGER(KIND=i_def), pointer :: wind_stencil_size(:,:)
    INTEGER(KIND=i_def), pointer :: wind_stencil_dofmap(:,:,:,:)
    TYPE(stencil_2D_dofmap_type), pointer :: wind_stencil_map

    nullify( m3_inv_local_stencil, set_counts_field_dummy_data, lookup_field_dummy_data, &
             set_counts_field_data, lookup_field_data, wind_data, advective_increment_data, &
             map_adspc1_lookup_field, map_adspc2_set_counts_field, map_any_w2, map_w3, &
             mesh, &
             set_counts_field_dummy_stencil_size, set_counts_field_dummy_stencil_dofmap, set_counts_field_dummy_stencil_map, &
             lookup_field_dummy_stencil_size, lookup_field_dummy_stencil_dofmap, lookup_field_dummy_stencil_map, &
             wind_stencil_size, wind_stencil_dofmap, wind_stencil_map )

    !
    ! Initialise field and/or operator proxies
    !
    advective_increment_proxy = advective_increment%get_proxy()
    advective_increment_data => advective_increment_proxy%data
    wind_proxy = wind%get_proxy()
    wind_data => wind_proxy%data
    m3_inv_proxy = m3_inv%get_proxy()
    m3_inv_local_stencil => m3_inv_proxy%local_stencil
    lookup_field_proxy = lookup_field%get_proxy()
    lookup_field_data => lookup_field_proxy%data
    set_counts_field_proxy = set_counts_field%get_proxy()
    set_counts_field_data => set_counts_field_proxy%data
    lookup_field_dummy_proxy = lookup_field_dummy%get_proxy()
    lookup_field_dummy_data => lookup_field_dummy_proxy%data
    set_counts_field_dummy_proxy = set_counts_field_dummy%get_proxy()
    set_counts_field_dummy_data => set_counts_field_dummy_proxy%data
    !
    ! Initialise number of layers
    !
    nlayers_advective_increment = advective_increment_proxy%vspace%get_nlayers()
    !
    ! Create a mesh object
    !
    mesh => advective_increment_proxy%vspace%get_mesh()
    max_halo_depth_mesh = mesh%get_halo_depth()
    !
    ! Initialise stencil dofmaps
    !
    wind_stencil_map => wind_proxy%vspace%get_stencil_2D_dofmap(STENCIL_2D_CROSS,stencil_extent)
    wind_max_branch_length = stencil_extent + 1_i_def
    wind_stencil_dofmap => wind_stencil_map%get_whole_dofmap()
    wind_stencil_size => wind_stencil_map%get_stencil_sizes()
    lookup_field_dummy_stencil_map => lookup_field_dummy_proxy%vspace%get_stencil_2D_dofmap(STENCIL_2D_CROSS,stencil_extent)
    lookup_field_dummy_max_branch_length = stencil_extent + 1_i_def
    lookup_field_dummy_stencil_dofmap => lookup_field_dummy_stencil_map%get_whole_dofmap()
    lookup_field_dummy_stencil_size => lookup_field_dummy_stencil_map%get_stencil_sizes()
    set_counts_field_dummy_stencil_map => &
&set_counts_field_dummy_proxy%vspace%get_stencil_2D_dofmap(STENCIL_2D_CROSS,stencil_extent)
    set_counts_field_dummy_max_branch_length = stencil_extent + 1_i_def
    set_counts_field_dummy_stencil_dofmap => set_counts_field_dummy_stencil_map%get_whole_dofmap()
    set_counts_field_dummy_stencil_size => set_counts_field_dummy_stencil_map%get_stencil_sizes()
    !
    ! Look-up dofmaps for each function space
    !
    map_w3 => advective_increment_proxy%vspace%get_whole_dofmap()
    map_any_w2 => wind_proxy%vspace%get_whole_dofmap()
    map_adspc1_lookup_field => lookup_field_proxy%vspace%get_whole_dofmap()
    map_adspc2_set_counts_field => set_counts_field_proxy%vspace%get_whole_dofmap()
    !
    ! Initialise number of DoFs for w3
    !
    ndf_w3 = advective_increment_proxy%vspace%get_ndf()
    undf_w3 = advective_increment_proxy%vspace%get_undf()
    !
    ! Initialise number of DoFs for any_w2
    !
    ndf_any_w2 = wind_proxy%vspace%get_ndf()
    undf_any_w2 = wind_proxy%vspace%get_undf()
    !
    ! Initialise number of DoFs for adspc1_lookup_field
    !
    ndf_adspc1_lookup_field = lookup_field_proxy%vspace%get_ndf()
    undf_adspc1_lookup_field = lookup_field_proxy%vspace%get_undf()
    !
    ! Initialise number of DoFs for adspc2_set_counts_field
    !
    ndf_adspc2_set_counts_field = set_counts_field_proxy%vspace%get_ndf()
    undf_adspc2_set_counts_field = set_counts_field_proxy%vspace%get_undf()
    !
    ! Set-up all of the loop bounds
    !
    loop0_start = 1
    loop0_stop = mesh%get_last_halo_cell(loop_halo_depth)
    !
    ! Call kernels and communication routines
    !

    IF (advective_increment_proxy%is_dirty(depth=loop_halo_depth)) THEN
      CALL advective_increment_proxy%halo_exchange(depth=loop_halo_depth)
    END IF
    IF (wind_proxy%is_dirty(depth=loop_halo_depth + stencil_extent)) THEN
      CALL wind_proxy%halo_exchange(depth=loop_halo_depth + stencil_extent)
    END IF
    IF (lookup_field_proxy%is_dirty(depth=loop_halo_depth + stencil_extent)) THEN
      CALL lookup_field_proxy%halo_exchange(depth=loop_halo_depth + stencil_extent)
    END IF
    IF (set_counts_field_proxy%is_dirty(depth=loop_halo_depth + stencil_extent)) THEN
      CALL set_counts_field_proxy%halo_exchange(depth=loop_halo_depth + stencil_extent)
    END IF
    DO cell = loop0_start, loop0_stop, 1
      CALL gen_w3h_adv_upd_lookup_code(cell, nlayers_advective_increment, advective_increment_data, wind_data, &
&wind_stencil_size(:,cell), wind_max_branch_length, wind_stencil_dofmap(:,:,:,cell), m3_inv_proxy%ncell_3d, &
&m3_inv_local_stencil, lookup_field_data, set_counts_field_data, &
&lookup_field_dummy_data, lookup_field_dummy_stencil_size(:,cell), lookup_field_dummy_max_branch_length, &
&lookup_field_dummy_stencil_dofmap(:,:,:,cell), set_counts_field_dummy_data, set_counts_field_dummy_stencil_size(:,cell), &
&set_counts_field_dummy_max_branch_length, set_counts_field_dummy_stencil_dofmap(:,:,:,cell), nsets_max, nindices, ndf_w3, &
&undf_w3, map_w3(:,cell), ndf_any_w2, undf_any_w2, map_any_w2(:,cell), ndf_adspc1_lookup_field, undf_adspc1_lookup_field, &
&map_adspc1_lookup_field(:,cell), ndf_adspc2_set_counts_field, undf_adspc2_set_counts_field, map_adspc2_set_counts_field(:,cell))
    END DO
    !
    ! Set halos dirty/clean for fields modified in the above loop
    !
    CALL lookup_field_proxy%set_dirty()
    CALL lookup_field_proxy%set_clean(loop_halo_depth)
    CALL set_counts_field_proxy%set_dirty()
    CALL set_counts_field_proxy%set_clean(loop_halo_depth)
    !
    !
  END SUBROUTINE invoke_gen_w3h_adv_upd_lookup_kernel

  SUBROUTINE invoke_gen_apply_helmholtz_op_lookup_kernel(dummy_w3_big_halo, lookup_field, set_counts_field, lookup_field_dummy, &
&set_counts_field_dummy, nindices, stencil_extent, loop_halo_depth)
    USE gen_apply_helmholtz_op_lookup_kernel_mod, ONLY: gen_apply_helmholtz_op_lookup_code
    USE mesh_mod, ONLY: mesh_type
    USE stencil_2D_dofmap_mod, ONLY: stencil_2D_dofmap_type, STENCIL_2D_CROSS
    INTEGER(KIND=i_def), intent(in) :: nindices
    TYPE(r_solver_field_type), intent(in) :: dummy_w3_big_halo
    TYPE(integer_field_type), intent(in) :: lookup_field, set_counts_field, lookup_field_dummy, set_counts_field_dummy
    INTEGER(KIND=i_def), intent(in) :: stencil_extent
    INTEGER, intent(in) :: loop_halo_depth
    INTEGER(KIND=i_def) cell
    INTEGER(KIND=i_def) loop0_start, loop0_stop
    INTEGER(KIND=i_def) nlayers_dummy_w3_big_halo
    INTEGER(KIND=i_def), pointer, dimension(:) :: set_counts_field_dummy_data
    INTEGER(KIND=i_def), pointer, dimension(:) :: lookup_field_dummy_data
    INTEGER(KIND=i_def), pointer, dimension(:) :: set_counts_field_data
    INTEGER(KIND=i_def), pointer, dimension(:) :: lookup_field_data
    TYPE(integer_field_proxy_type) lookup_field_proxy, set_counts_field_proxy, lookup_field_dummy_proxy, &
&set_counts_field_dummy_proxy
    REAL(KIND=r_solver), pointer, dimension(:) :: dummy_w3_big_halo_data
    TYPE(r_solver_field_proxy_type) dummy_w3_big_halo_proxy
    INTEGER(KIND=i_def), pointer :: map_adspc1_lookup_field(:,:), map_adspc2_set_counts_field(:,:), &
&map_w3(:,:)
    INTEGER(KIND=i_def) ndf_w3, undf_w3, ndf_adspc1_lookup_field, undf_adspc1_lookup_field, ndf_adspc2_set_counts_field, &
&undf_adspc2_set_counts_field
    INTEGER(KIND=i_def) max_halo_depth_mesh
    TYPE(mesh_type), pointer :: mesh
    INTEGER(KIND=i_def) set_counts_field_dummy_max_branch_length
    INTEGER(KIND=i_def), pointer :: set_counts_field_dummy_stencil_size(:,:)
    INTEGER(KIND=i_def), pointer :: set_counts_field_dummy_stencil_dofmap(:,:,:,:)
    TYPE(stencil_2D_dofmap_type), pointer :: set_counts_field_dummy_stencil_map
    INTEGER(KIND=i_def) lookup_field_dummy_max_branch_length
    INTEGER(KIND=i_def), pointer :: lookup_field_dummy_stencil_size(:,:)
    INTEGER(KIND=i_def), pointer :: lookup_field_dummy_stencil_dofmap(:,:,:,:)
    TYPE(stencil_2D_dofmap_type), pointer :: lookup_field_dummy_stencil_map
    INTEGER(KIND=i_def) dummy_w3_big_halo_max_branch_length
    INTEGER(KIND=i_def), pointer :: dummy_w3_big_halo_stencil_size(:,:)
    INTEGER(KIND=i_def), pointer :: dummy_w3_big_halo_stencil_dofmap(:,:,:,:)
    TYPE(stencil_2D_dofmap_type), pointer :: dummy_w3_big_halo_stencil_map
    nullify( set_counts_field_dummy_data, lookup_field_dummy_data, set_counts_field_data, &
             lookup_field_data, dummy_w3_big_halo_data, map_adspc1_lookup_field, &
             map_adspc2_set_counts_field, mesh, set_counts_field_dummy_stencil_size, &
             set_counts_field_dummy_stencil_dofmap, set_counts_field_dummy_stencil_map, &
             lookup_field_dummy_stencil_size, lookup_field_dummy_stencil_dofmap, &
             lookup_field_dummy_stencil_map, dummy_w3_big_halo_stencil_size, &
             dummy_w3_big_halo_stencil_dofmap, dummy_w3_big_halo_stencil_map )
    !
    ! Initialise field and/or operator proxies
    !
    dummy_w3_big_halo_proxy = dummy_w3_big_halo%get_proxy()
    dummy_w3_big_halo_data => dummy_w3_big_halo_proxy%data
    lookup_field_proxy = lookup_field%get_proxy()
    lookup_field_data => lookup_field_proxy%data
    set_counts_field_proxy = set_counts_field%get_proxy()
    set_counts_field_data => set_counts_field_proxy%data
    lookup_field_dummy_proxy = lookup_field_dummy%get_proxy()
    lookup_field_dummy_data => lookup_field_dummy_proxy%data
    set_counts_field_dummy_proxy = set_counts_field_dummy%get_proxy()
    set_counts_field_dummy_data => set_counts_field_dummy_proxy%data
    !
    ! Initialise number of layers
    !
    nlayers_dummy_w3_big_halo = dummy_w3_big_halo_proxy%vspace%get_nlayers()
    !
    ! Create a mesh object
    !
    mesh => dummy_w3_big_halo_proxy%vspace%get_mesh()
    max_halo_depth_mesh = mesh%get_halo_depth()
    !
    ! Initialise stencil dofmaps
    !
    dummy_w3_big_halo_stencil_map => dummy_w3_big_halo_proxy%vspace%get_stencil_2D_dofmap(STENCIL_2D_CROSS,stencil_extent)
    dummy_w3_big_halo_max_branch_length = stencil_extent + 1_i_def
    dummy_w3_big_halo_stencil_dofmap => dummy_w3_big_halo_stencil_map%get_whole_dofmap()
    dummy_w3_big_halo_stencil_size => dummy_w3_big_halo_stencil_map%get_stencil_sizes()
    lookup_field_dummy_stencil_map => lookup_field_dummy_proxy%vspace%get_stencil_2D_dofmap(STENCIL_2D_CROSS,stencil_extent)
    lookup_field_dummy_max_branch_length = stencil_extent + 1_i_def
    lookup_field_dummy_stencil_dofmap => lookup_field_dummy_stencil_map%get_whole_dofmap()
    lookup_field_dummy_stencil_size => lookup_field_dummy_stencil_map%get_stencil_sizes()
    set_counts_field_dummy_stencil_map => &
&set_counts_field_dummy_proxy%vspace%get_stencil_2D_dofmap(STENCIL_2D_CROSS,stencil_extent)
    set_counts_field_dummy_max_branch_length = stencil_extent + 1_i_def
    set_counts_field_dummy_stencil_dofmap => set_counts_field_dummy_stencil_map%get_whole_dofmap()
    set_counts_field_dummy_stencil_size => set_counts_field_dummy_stencil_map%get_stencil_sizes()
    !
    ! Look-up dofmaps for each function space
    !
    map_w3 => dummy_w3_big_halo_proxy%vspace%get_whole_dofmap()
    map_adspc1_lookup_field => lookup_field_proxy%vspace%get_whole_dofmap()
    map_adspc2_set_counts_field => set_counts_field_proxy%vspace%get_whole_dofmap()
    !
    ! Initialise number of DoFs for w3
    !
    ndf_w3 = dummy_w3_big_halo_proxy%vspace%get_ndf()
    undf_w3 = dummy_w3_big_halo_proxy%vspace%get_undf()
    !
    ! Initialise number of DoFs for adspc1_lookup_field
    !
    ndf_adspc1_lookup_field = lookup_field_proxy%vspace%get_ndf()
    undf_adspc1_lookup_field = lookup_field_proxy%vspace%get_undf()
    !
    ! Initialise number of DoFs for adspc2_set_counts_field
    !
    ndf_adspc2_set_counts_field = set_counts_field_proxy%vspace%get_ndf()
    undf_adspc2_set_counts_field = set_counts_field_proxy%vspace%get_undf()
    !
    ! Set-up all of the loop bounds
    !
    loop0_start = 1
    loop0_stop = mesh%get_last_halo_cell(loop_halo_depth)
    !
    ! Call kernels and communication routines
    !
    IF (lookup_field_proxy%is_dirty(depth=loop_halo_depth + stencil_extent)) THEN
      CALL lookup_field_proxy%halo_exchange(depth=loop_halo_depth + stencil_extent)
    END IF
    IF (set_counts_field_proxy%is_dirty(depth=loop_halo_depth + stencil_extent)) THEN
      CALL set_counts_field_proxy%halo_exchange(depth=loop_halo_depth + stencil_extent)
    END IF
    DO cell = loop0_start, loop0_stop, 1
      CALL gen_apply_helmholtz_op_lookup_code(nlayers_dummy_w3_big_halo, dummy_w3_big_halo_data, &
&dummy_w3_big_halo_stencil_size(:,cell), dummy_w3_big_halo_max_branch_length, dummy_w3_big_halo_stencil_dofmap(:,:,:,cell), &
&lookup_field_data, set_counts_field_data, lookup_field_dummy_data, lookup_field_dummy_stencil_size(:,cell), &
&lookup_field_dummy_max_branch_length, lookup_field_dummy_stencil_dofmap(:,:,:,cell), set_counts_field_dummy_data, &
&set_counts_field_dummy_stencil_size(:,cell), set_counts_field_dummy_max_branch_length, &
&set_counts_field_dummy_stencil_dofmap(:,:,:,cell), nindices, ndf_w3, undf_w3, map_w3(:,cell), ndf_adspc1_lookup_field, &
&undf_adspc1_lookup_field, map_adspc1_lookup_field(:,cell), ndf_adspc2_set_counts_field, undf_adspc2_set_counts_field, &
&map_adspc2_set_counts_field(:,cell))
    END DO
    !
    ! Set halos dirty/clean for fields modified in the above loop
    !
    CALL lookup_field_proxy%set_dirty()
    CALL lookup_field_proxy%set_clean(loop_halo_depth)
    CALL set_counts_field_proxy%set_dirty()
    CALL set_counts_field_proxy%set_clean(loop_halo_depth)
    !
    !
  END SUBROUTINE invoke_gen_apply_helmholtz_op_lookup_kernel

end module psykal_lite_gen_lookup_tables_psy_mod
