.amdhsa_code_object_version 6

.text
.globl    kernel_func                       ; * kernel function name.
.type     kernel_func,@function
.p2align  8
kernel_func:
	.long 0xbf9f0000

.section  .rodata,"a",@progbits
.p2align  6, 0x0
.amdhsa_kernel kernel_func
	.amdhsa_group_segment_fixed_size 327680 ; * LDS size in byte
	.amdhsa_kernarg_size 80                 ; * kernel args size in byte
	.amdhsa_next_free_vgpr 64               ; * max VGPR index plus one.
	.amdhsa_next_free_sgpr 64               ; * max SGPR index plus one.
	.amdhsa_system_sgpr_workgroup_id_x 1    ;   ENABLE_SGPR_WORKGROUP_ID_X 
	.amdhsa_system_sgpr_workgroup_id_y 1    ;   ENABLE_SGPR_WORKGROUP_ID_Y
	.amdhsa_system_sgpr_workgroup_id_z 0    ;   ENABLE_SGPR_WORKGROUP_ID_z
	.amdhsa_system_vgpr_workitem_id 2       ;   0=Set work-item X dimension ID; 1=Set work-item X and Y dimensions ID; 2=Set work-item X, Y and Z dimensions ID
	.amdhsa_user_sgpr_count 2
	.amdhsa_user_sgpr_kernarg_segment_ptr 1
	.amdhsa_inst_pref_size 255
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_named_barrier_count 0
	.amdhsa_round_robin_scheduling 0
	.amdhsa_wavefront_size32 1
	; default -----------------------------------
	;.amdhsa_enable_private_segment 0
	;.amdhsa_private_segment_fixed_size 0
	;.amdhsa_user_sgpr_dispatch_ptr 0
	;.amdhsa_user_sgpr_queue_ptr 0
	;.amdhsa_user_sgpr_dispatch_id 0
	;.amdhsa_user_sgpr_private_segment_size 0
	;.amdhsa_uses_dynamic_stack 0
	;.amdhsa_system_sgpr_workgroup_info 0
	;.amdhsa_reserve_vcc 1
	;.amdhsa_float_denorm_mode_32 0
	;.amdhsa_float_round_mode_32 0
	;.amdhsa_float_round_mode_16_64 0
	;.amdhsa_float_denorm_mode_16_64 3 
	;.amdhsa_fp16_overflow 0
	;.amdhsa_exception_fp_ieee_invalid_op 0
	;.amdhsa_exception_fp_denorm_src 0
	;.amdhsa_exception_fp_ieee_div_zero 0
	;.amdhsa_exception_fp_ieee_overflow 0
	;.amdhsa_exception_fp_ieee_underflow 0
	;.amdhsa_exception_fp_ieee_inexact 0
	;.amdhsa_exception_int_div_zero 0
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
    - {.value_kind: global_buffer, .offset:  0, .size: 8, .actual_access: read_only,  .address_space: global}  
    - {.value_kind: global_buffer, .offset: 16, .size: 8, .actual_access: read_only,  .address_space: global}
    - {.value_kind: global_buffer, .offset: 32, .size: 8, .actual_access: read_write, .address_space: global}
    - {.value_kind: by_value,      .offset: 48, .size: 4}        
    - {.value_kind: by_value,      .offset: 64, .size: 4}
    .kernarg_segment_align:      8
    .kernarg_segment_size:       80     ; * kernel args size in byte
    .sgpr_count:                 64     ; * SGPR number
    .vgpr_count:                 64     ; * VGPR number
    .group_segment_fixed_size:   327680 ; * LDS size in byte
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size:    1024
    .wavefront_size: 32
    .uses_dynamic_stack: false
    .uniform_work_group_size:  1
    .workgroup_processor_mode: 1
    ;.cluster_dims: [ ?, ?, ? ] ; co v6 only
    .name:       kernel_func
    .symbol:     kernel_func.kd
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [ 1, 2 ]
...
.end_amdgpu_metadata
