# sum8(gpr) -> vec
# frame: 0 bytes aligned to 0, saves []
# xs=rdi v=ymm0 sum=ymm0 high=ymm1 pair=ymm1
    vmovups ymm0, ymmword ptr [rdi]
    # the lanes
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vmovhlps xmm1, xmm0, xmm0
    vaddps xmm0, xmm0, xmm1
    vmovshdup xmm1, xmm0
    vaddss xmm0, xmm0, xmm1
    vzeroupper
    ret
