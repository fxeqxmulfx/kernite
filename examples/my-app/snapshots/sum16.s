# sum16(gpr) -> vec
# needs: avx
# frame: 0 bytes aligned to 0, saves [], and 8 more so that it calls at a multiple of 16
# xs=rdi low=ymm2,ymm0 high=ymm0
    sub rsp, 8
    call {f0}
    vmovaps xmm2, xmm0
    add rdi, 32
    call {f0}
    vaddss xmm0, xmm2, xmm0
    add rsp, 8
    ret
# f0: sum8
