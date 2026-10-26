# dot::<4>(gpr, gpr, gpr) -> vec
# frame: 0 bytes aligned to 0, saves []
# x=rdi y=rsi n=rdx acc0=ymm0 acc1=ymm1 acc2=ymm2
# acc3=ymm3 xs=ymm4 sum=ymm0 high=ymm1 pair=ymm1 zero=ymm0
    test rdx, rdx
    jz 3f
    vxorps xmm0, xmm0, xmm0
    vxorps xmm1, xmm1, xmm1
    vxorps xmm2, xmm2, xmm2
    vxorps xmm3, xmm3, xmm3
2:
    vmovups ymm4, ymmword ptr [rdi]
    vfmadd231ps ymm0, ymm4, ymmword ptr [rsi]
    vmovups ymm4, ymmword ptr [rdi + 32]
    vfmadd231ps ymm1, ymm4, ymmword ptr [rsi + 32]
    vmovups ymm4, ymmword ptr [rdi + 64]
    vfmadd231ps ymm2, ymm4, ymmword ptr [rsi + 64]
    vmovups ymm4, ymmword ptr [rdi + 96]
    vfmadd231ps ymm3, ymm4, ymmword ptr [rsi + 96]
    add rdi, 128
    add rsi, 128
    sub rdx, 32
    jnz 2b
    # the accumulators, in order
    vaddps ymm0, ymm0, ymm1
    vaddps ymm0, ymm0, ymm2
    vaddps ymm0, ymm0, ymm3
    # the lanes
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vmovhlps xmm1, xmm0, xmm0
    vaddps xmm0, xmm0, xmm1
    vmovshdup xmm1, xmm0
    vaddss xmm0, xmm0, xmm1
    vzeroupper
    ret
3:
    vxorps xmm0, xmm0, xmm0
    vzeroupper
    ret
