# lm_bwd_w(gpr, gpr, gpr, gpr) -> ()
# frame: 0 bytes aligned to 0, saves []
# g=rdi x=rsi dy=rdx n=rcx i=rax acc0=ymm0
# acc1=ymm1 acc2=ymm2 acc3=ymm3 xt=r8 dyt=r9 left=r10
# b=ymm4
    test rcx, rcx
    jz 4f
    mov rax, 16
2:
    vmovups ymm0, ymmword ptr [rdi]
    vmovups ymm1, ymmword ptr [rdi + 32]
    vmovups ymm2, ymmword ptr [rdi + 64]
    vmovups ymm3, ymmword ptr [rdi + 96]
    mov r8, rsi
    mov r9, rdx
    mov r10, rcx
3:
    vbroadcastss ymm4, dword ptr [r8]
    vfmadd231ps ymm0, ymm4, ymmword ptr [r9]
    vfmadd231ps ymm1, ymm4, ymmword ptr [r9 + 32]
    vfmadd231ps ymm2, ymm4, ymmword ptr [r9 + 64]
    vfmadd231ps ymm3, ymm4, ymmword ptr [r9 + 96]
    add r8, 64
    add r9, 128
    dec r10
    jnz 3b
    vmovups ymmword ptr [rdi], ymm0
    vmovups ymmword ptr [rdi + 32], ymm1
    vmovups ymmword ptr [rdi + 64], ymm2
    vmovups ymmword ptr [rdi + 96], ymm3
    add rdi, 128
    add rsi, 4
    dec rax
    jnz 2b
4:
    vzeroupper
    ret
