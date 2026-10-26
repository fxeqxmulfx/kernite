# embed_bwd(gpr, gpr, gpr, gpr) -> ()
# needs: avx
# frame: 0 bytes aligned to 0, saves []
# g=rdi dx0=rsi tokens=rdx n=rcx at=rax tok=r8
# d=ymm0 s=ymm1
    test rcx, rcx
    jz 3f
    lea rax, [rdi + 1728]
2:
    mov r8d, dword ptr [rdx]
    shl r8, 6
    vmovups ymm0, ymmword ptr [rsi]
    vaddps ymm1, ymm0, ymmword ptr [rdi + r8*1]
    vmovups ymmword ptr [rdi + r8*1], ymm1
    vaddps ymm1, ymm0, ymmword ptr [rax]
    vmovups ymmword ptr [rax], ymm1
    vmovups ymm0, ymmword ptr [rsi + 32]
    vaddps ymm1, ymm0, ymmword ptr [rdi + r8*1 + 32]
    vmovups ymmword ptr [rdi + r8*1 + 32], ymm1
    vaddps ymm1, ymm0, ymmword ptr [rax + 32]
    vmovups ymmword ptr [rax + 32], ymm1
    add rdx, 4
    add rsi, 64
    add rax, 64
    dec rcx
    jnz 2b
3:
    vzeroupper
    ret
