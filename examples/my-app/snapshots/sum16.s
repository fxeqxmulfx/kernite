# sum16(gpr) -> vec
# needs: avx
# frame: 0 bytes aligned to 0, saves [], and 8 more so that it calls at a multiple of 16
# xs=rax,rdi low=ymm2,ymm0 high=ymm0
    sub rsp, 8
    mov rax, rdi
    call {f0}
    vmovaps xmm2, xmm0
    add rax, 32
    mov rdi, rax
    call {f0}
    vaddss xmm0, xmm2, xmm0
    add rsp, 8
    ret
# f0: sum8
