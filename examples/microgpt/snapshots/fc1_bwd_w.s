# fc1_bwd_w(gpr, gpr, gpr, gpr) -> ()
# needs: avx, fma
# frame: 0 bytes aligned to 0, saves []
# g=rdi x=rsi dy=rdx n=rcx i=rax acc0=ymm0
# acc1=ymm1 acc2=ymm2 acc3=ymm3 acc4=ymm4 acc5=ymm5 acc6=ymm6
# acc7=ymm7 xt=r8 dyt=r9 left=r10 b=ymm8
    test rcx, rcx
    jz 4f
    mov rax, 16
2:
    vmovups ymm0, ymmword ptr [rdi]
    vmovups ymm1, ymmword ptr [rdi + 32]
    vmovups ymm2, ymmword ptr [rdi + 64]
    vmovups ymm3, ymmword ptr [rdi + 96]
    vmovups ymm4, ymmword ptr [rdi + 128]
    vmovups ymm5, ymmword ptr [rdi + 160]
    vmovups ymm6, ymmword ptr [rdi + 192]
    vmovups ymm7, ymmword ptr [rdi + 224]
    mov r8, rsi
    mov r9, rdx
    mov r10, rcx
3:
    vbroadcastss ymm8, dword ptr [r8]
    vfmadd231ps ymm0, ymm8, ymmword ptr [r9]
    vfmadd231ps ymm1, ymm8, ymmword ptr [r9 + 32]
    vfmadd231ps ymm2, ymm8, ymmword ptr [r9 + 64]
    vfmadd231ps ymm3, ymm8, ymmword ptr [r9 + 96]
    vfmadd231ps ymm4, ymm8, ymmword ptr [r9 + 128]
    vfmadd231ps ymm5, ymm8, ymmword ptr [r9 + 160]
    vfmadd231ps ymm6, ymm8, ymmword ptr [r9 + 192]
    vfmadd231ps ymm7, ymm8, ymmword ptr [r9 + 224]
    add r8, 64
    add r9, 256
    dec r10
    jnz 3b
    vmovups ymmword ptr [rdi], ymm0
    vmovups ymmword ptr [rdi + 32], ymm1
    vmovups ymmword ptr [rdi + 64], ymm2
    vmovups ymmword ptr [rdi + 96], ymm3
    vmovups ymmword ptr [rdi + 128], ymm4
    vmovups ymmword ptr [rdi + 160], ymm5
    vmovups ymmword ptr [rdi + 192], ymm6
    vmovups ymmword ptr [rdi + 224], ymm7
    add rdi, 256
    add rsi, 4
    dec rax
    jnz 2b
4:
    vzeroupper
    ret
