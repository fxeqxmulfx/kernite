# fc2_bwd_w(gpr, gpr, gpr, gpr) -> ()
# frame: 0 bytes aligned to 0, saves []
# g=rdi x=rsi dy=rdx n=rcx i=rax acc0=ymm0
# acc1=ymm1 xt=r8 dyt=r9 left=r10 b=ymm2
    test rcx, rcx
    jz 4f
    mov rax, 64
2:
    vmovups ymm0, ymmword ptr [rdi]
    vmovups ymm1, ymmword ptr [rdi + 32]
    mov r8, rsi
    mov r9, rdx
    mov r10, rcx
3:
    vbroadcastss ymm2, dword ptr [r8]
    vfmadd231ps ymm0, ymm2, ymmword ptr [r9]
    vfmadd231ps ymm1, ymm2, ymmword ptr [r9 + 32]
    add r8, 256
    add r9, 64
    dec r10
    jnz 3b
    vmovups ymmword ptr [rdi], ymm0
    vmovups ymmword ptr [rdi + 32], ymm1
    add rdi, 64
    add rsi, 4
    dec rax
    jnz 2b
4:
    vzeroupper
    ret
