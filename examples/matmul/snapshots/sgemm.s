# sgemm(gpr, gpr, gpr, gpr, gpr, gpr) -> ()
# needs: avx, fma
# frame: 0 bytes aligned to 0, saves [rbx]
# c=rdi a=rsi b=rdx m=rcx n=r8 k=r9
# j=rax acc=ymm0 x=r10 y=r11 l=rbx v=ymm1
    push rbx
    test rcx, rcx
    jz 6f
    test r8, r8
    jz 6f
2:
    mov eax, 0
3:
    vxorps xmm0, xmm0, xmm0
    test r9, r9
    jz 5f
    # down column `j` of `b`, along row `a`
    mov r10, rsi
    lea r11, [rdx + rax*4]
    mov rbx, r9
4:
    vbroadcastss ymm1, dword ptr [r10]
    vfmadd231ps ymm0, ymm1, ymmword ptr [r11]
    add r10, 4
    lea r11, [r11 + r8*4]
    dec rbx
    jnz 4b
5:
    vmovups ymmword ptr [rdi + rax*4], ymm0
    add rax, 8
    cmp rax, r8
    jb 3b
    lea rsi, [rsi + r9*4]
    lea rdi, [rdi + r8*4]
    dec rcx
    jnz 2b
6:
    pop rbx
    vzeroupper
    ret
