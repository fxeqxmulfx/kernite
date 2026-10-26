# dot::<16>(gpr, gpr, gpr) -> vec
# needs: avx, fma
# frame: 64 bytes aligned to 32, saves []
# acc15, spilled: 32 bytes at rsp
# acc14, spilled: 32 bytes at rsp + 32
# x=rdi y=rsi n=rdx acc0=ymm0 acc1=ymm1 acc2=ymm2
# acc3=ymm3 acc4=ymm4 acc5=ymm5 acc6=ymm6 acc7=ymm7 acc8=ymm8
# acc9=ymm9 acc10=ymm10 acc11=ymm11 acc12=ymm12 acc13=ymm13 xs=ymm14
# sum=ymm0 high=ymm1 pair=ymm1 zero=ymm0 acc15=ymm14,ymm15,ymm1 acc14=ymm14,ymm15,ymm1
    push rbp
    mov rbp, rsp
    sub rsp, 64
    and rsp, -32
    test rdx, rdx
    jz 3f
    vxorps xmm0, xmm0, xmm0
    vxorps xmm1, xmm1, xmm1
    vxorps xmm2, xmm2, xmm2
    vxorps xmm3, xmm3, xmm3
    vxorps xmm4, xmm4, xmm4
    vxorps xmm5, xmm5, xmm5
    vxorps xmm6, xmm6, xmm6
    vxorps xmm7, xmm7, xmm7
    vxorps xmm8, xmm8, xmm8
    vxorps xmm9, xmm9, xmm9
    vxorps xmm10, xmm10, xmm10
    vxorps xmm11, xmm11, xmm11
    vxorps xmm12, xmm12, xmm12
    vxorps xmm13, xmm13, xmm13
    vxorps xmm14, xmm14, xmm14
    # acc14 spilled
    vmovups ymmword ptr [rsp + 32], ymm14
    vxorps xmm14, xmm14, xmm14
    # acc15 spilled
    vmovups ymmword ptr [rsp], ymm14
2:
    vmovups ymm14, ymmword ptr [rdi]
    vfmadd231ps ymm0, ymm14, ymmword ptr [rsi]
    vmovups ymm14, ymmword ptr [rdi + 32]
    vfmadd231ps ymm1, ymm14, ymmword ptr [rsi + 32]
    vmovups ymm14, ymmword ptr [rdi + 64]
    vfmadd231ps ymm2, ymm14, ymmword ptr [rsi + 64]
    vmovups ymm14, ymmword ptr [rdi + 96]
    vfmadd231ps ymm3, ymm14, ymmword ptr [rsi + 96]
    vmovups ymm14, ymmword ptr [rdi + 128]
    vfmadd231ps ymm4, ymm14, ymmword ptr [rsi + 128]
    vmovups ymm14, ymmword ptr [rdi + 160]
    vfmadd231ps ymm5, ymm14, ymmword ptr [rsi + 160]
    vmovups ymm14, ymmword ptr [rdi + 192]
    vfmadd231ps ymm6, ymm14, ymmword ptr [rsi + 192]
    vmovups ymm14, ymmword ptr [rdi + 224]
    vfmadd231ps ymm7, ymm14, ymmword ptr [rsi + 224]
    vmovups ymm14, ymmword ptr [rdi + 256]
    vfmadd231ps ymm8, ymm14, ymmword ptr [rsi + 256]
    vmovups ymm14, ymmword ptr [rdi + 288]
    vfmadd231ps ymm9, ymm14, ymmword ptr [rsi + 288]
    vmovups ymm14, ymmword ptr [rdi + 320]
    vfmadd231ps ymm10, ymm14, ymmword ptr [rsi + 320]
    vmovups ymm14, ymmword ptr [rdi + 352]
    vfmadd231ps ymm11, ymm14, ymmword ptr [rsi + 352]
    vmovups ymm14, ymmword ptr [rdi + 384]
    vfmadd231ps ymm12, ymm14, ymmword ptr [rsi + 384]
    vmovups ymm14, ymmword ptr [rdi + 416]
    vfmadd231ps ymm13, ymm14, ymmword ptr [rsi + 416]
    vmovups ymm14, ymmword ptr [rdi + 448]
    # acc14 reloaded
    vmovups ymm15, ymmword ptr [rsp + 32]
    vfmadd231ps ymm15, ymm14, ymmword ptr [rsi + 448]
    # acc14 spilled
    vmovups ymmword ptr [rsp + 32], ymm15
    vmovups ymm14, ymmword ptr [rdi + 480]
    # acc15 reloaded
    vmovups ymm15, ymmword ptr [rsp]
    vfmadd231ps ymm15, ymm14, ymmword ptr [rsi + 480]
    # acc15 spilled
    vmovups ymmword ptr [rsp], ymm15
    add rdi, 512
    add rsi, 512
    sub rdx, 128
    jnz 2b
    # the accumulators, in order
    vaddps ymm0, ymm0, ymm1
    vaddps ymm0, ymm0, ymm2
    vaddps ymm0, ymm0, ymm3
    vaddps ymm0, ymm0, ymm4
    vaddps ymm0, ymm0, ymm5
    vaddps ymm0, ymm0, ymm6
    vaddps ymm0, ymm0, ymm7
    vaddps ymm0, ymm0, ymm8
    vaddps ymm0, ymm0, ymm9
    vaddps ymm0, ymm0, ymm10
    vaddps ymm0, ymm0, ymm11
    vaddps ymm0, ymm0, ymm12
    vaddps ymm0, ymm0, ymm13
    # acc14 reloaded
    vmovups ymm1, ymmword ptr [rsp + 32]
    vaddps ymm0, ymm0, ymm1
    # acc15 reloaded
    vmovups ymm1, ymmword ptr [rsp]
    vaddps ymm0, ymm0, ymm1
    # the lanes
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vmovhlps xmm1, xmm0, xmm0
    vaddps xmm0, xmm0, xmm1
    vmovshdup xmm1, xmm0
    vaddss xmm0, xmm0, xmm1
    mov rsp, rbp
    pop rbp
    vzeroupper
    ret
3:
    vxorps xmm0, xmm0, xmm0
    mov rsp, rbp
    pop rbp
    vzeroupper
    ret
