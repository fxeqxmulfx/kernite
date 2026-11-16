# sum16(gpr) -> vec
# needs: avx
# frame: 32 bytes aligned to 32, saves []
# low, spilled: 32 bytes at rsp
# xs=rax,rdi high=ymm0 low=ymm0,ymm1
    push rbp
    mov rbp, rsp
    sub rsp, 32
    and rsp, -32
    mov rax, rdi
    mov rdi, rax
    call {f0}
    # low spilled
    vmovups ymmword ptr [rsp], ymm0
    add rax, 32
    mov rdi, rax
    call {f0}
    # low reloaded
    vmovups ymm1, ymmword ptr [rsp]
    vaddss xmm0, xmm1, xmm0
    mov rsp, rbp
    pop rbp
    vzeroupper
    ret
# f0: sum8
