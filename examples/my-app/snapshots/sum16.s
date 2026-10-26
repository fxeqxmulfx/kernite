# sum16(gpr) -> vec
# needs: avx
# frame: 32 bytes aligned to 32, saves [rbx]
# low, spilled: 32 bytes at rsp
# xs=rbx,rdi high=ymm0 low=ymm0,ymm1
    push rbx
    push rbp
    mov rbp, rsp
    sub rsp, 32
    and rsp, -32
    mov rbx, rdi
    mov rdi, rbx
    call {f0}
    # low spilled
    vmovups ymmword ptr [rsp], ymm0
    add rbx, 32
    mov rdi, rbx
    call {f0}
    # low reloaded
    vmovups ymm1, ymmword ptr [rsp]
    vaddss xmm0, xmm1, xmm0
    mov rsp, rbp
    pop rbp
    pop rbx
    vzeroupper
    ret
# f0: sum8
