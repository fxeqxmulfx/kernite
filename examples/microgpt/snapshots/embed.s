# embed(gpr, gpr, gpr, gpr, gpr) -> ()
# needs: avx
# frame: 0 bytes aligned to 0, saves []
# x0=rdi p=rsi tokens=rdx t0=rcx n=r8 tok=rax
# at=r9 x=ymm0
    cmp rcx, r8
    jae 3f
2:
    mov eax, dword ptr [rdx + rcx*4]
    shl rax, 6
    mov r9, rcx
    shl r9, 6
    vmovups ymm0, ymmword ptr [rsi + rax*1]
    vaddps ymm0, ymm0, ymmword ptr [rsi + r9*1 + 1728]
    vmovups ymmword ptr [rdi + r9*1], ymm0
    vmovups ymm0, ymmword ptr [rsi + rax*1 + 32]
    vaddps ymm0, ymm0, ymmword ptr [rsi + r9*1 + 1760]
    vmovups ymmword ptr [rdi + r9*1 + 32], ymm0
    inc rcx
    cmp rcx, r8
    jb 2b
3:
    vzeroupper
    ret
