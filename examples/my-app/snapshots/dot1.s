# dot1(gpr, gpr, gpr) -> vec
# frame: 0 bytes aligned to 0, saves []
# x=rdi y=rsi n=rdx acc0=ymm0 xs=ymm1 high=ymm1
# sum=ymm0 pair=ymm1 zero=ymm0
    test rdx, rdx
    jz 3f
    vxorps xmm0, xmm0, xmm0
2:
    vmovups ymm1, ymmword ptr [rdi]
    vfmadd231ps ymm0, ymm1, ymmword ptr [rsi]
    add rdi, 32
    add rsi, 32
    sub rdx, 8
    jnz 2b
    # the accumulators, in order
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
