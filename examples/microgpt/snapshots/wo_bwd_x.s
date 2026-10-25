# wo_bwd_x(gpr, gpr, gpr, gpr) -> ()
# frame: 0 bytes aligned to 0, saves []
# dx=rdi dy=rsi w=rdx n=rcx wi=rax xi=r8
# left=r9 acc0=ymm0 acc1=ymm1 acc2=ymm2 acc3=ymm3 g=ymm4
    test rcx, rcx
    jz 4f
2:
    mov rax, rdx
    mov r8, rdi
    mov r9, 4
3:
    vmovss xmm0, dword ptr [r8]
    vmovss xmm1, dword ptr [r8 + 4]
    vmovss xmm2, dword ptr [r8 + 8]
    vmovss xmm3, dword ptr [r8 + 12]
    vmovss xmm4, dword ptr [rsi]
    vfmadd231ss xmm0, xmm4, dword ptr [rax]
    vfmadd231ss xmm1, xmm4, dword ptr [rax + 64]
    vfmadd231ss xmm2, xmm4, dword ptr [rax + 128]
    vfmadd231ss xmm3, xmm4, dword ptr [rax + 192]
    vmovss xmm4, dword ptr [rsi + 4]
    vfmadd231ss xmm0, xmm4, dword ptr [rax + 4]
    vfmadd231ss xmm1, xmm4, dword ptr [rax + 68]
    vfmadd231ss xmm2, xmm4, dword ptr [rax + 132]
    vfmadd231ss xmm3, xmm4, dword ptr [rax + 196]
    vmovss xmm4, dword ptr [rsi + 8]
    vfmadd231ss xmm0, xmm4, dword ptr [rax + 8]
    vfmadd231ss xmm1, xmm4, dword ptr [rax + 72]
    vfmadd231ss xmm2, xmm4, dword ptr [rax + 136]
    vfmadd231ss xmm3, xmm4, dword ptr [rax + 200]
    vmovss xmm4, dword ptr [rsi + 12]
    vfmadd231ss xmm0, xmm4, dword ptr [rax + 12]
    vfmadd231ss xmm1, xmm4, dword ptr [rax + 76]
    vfmadd231ss xmm2, xmm4, dword ptr [rax + 140]
    vfmadd231ss xmm3, xmm4, dword ptr [rax + 204]
    vmovss xmm4, dword ptr [rsi + 16]
    vfmadd231ss xmm0, xmm4, dword ptr [rax + 16]
    vfmadd231ss xmm1, xmm4, dword ptr [rax + 80]
    vfmadd231ss xmm2, xmm4, dword ptr [rax + 144]
    vfmadd231ss xmm3, xmm4, dword ptr [rax + 208]
    vmovss xmm4, dword ptr [rsi + 20]
    vfmadd231ss xmm0, xmm4, dword ptr [rax + 20]
    vfmadd231ss xmm1, xmm4, dword ptr [rax + 84]
    vfmadd231ss xmm2, xmm4, dword ptr [rax + 148]
    vfmadd231ss xmm3, xmm4, dword ptr [rax + 212]
    vmovss xmm4, dword ptr [rsi + 24]
    vfmadd231ss xmm0, xmm4, dword ptr [rax + 24]
    vfmadd231ss xmm1, xmm4, dword ptr [rax + 88]
    vfmadd231ss xmm2, xmm4, dword ptr [rax + 152]
    vfmadd231ss xmm3, xmm4, dword ptr [rax + 216]
    vmovss xmm4, dword ptr [rsi + 28]
    vfmadd231ss xmm0, xmm4, dword ptr [rax + 28]
    vfmadd231ss xmm1, xmm4, dword ptr [rax + 92]
    vfmadd231ss xmm2, xmm4, dword ptr [rax + 156]
    vfmadd231ss xmm3, xmm4, dword ptr [rax + 220]
    vmovss xmm4, dword ptr [rsi + 32]
    vfmadd231ss xmm0, xmm4, dword ptr [rax + 32]
    vfmadd231ss xmm1, xmm4, dword ptr [rax + 96]
    vfmadd231ss xmm2, xmm4, dword ptr [rax + 160]
    vfmadd231ss xmm3, xmm4, dword ptr [rax + 224]
    vmovss xmm4, dword ptr [rsi + 36]
    vfmadd231ss xmm0, xmm4, dword ptr [rax + 36]
    vfmadd231ss xmm1, xmm4, dword ptr [rax + 100]
    vfmadd231ss xmm2, xmm4, dword ptr [rax + 164]
    vfmadd231ss xmm3, xmm4, dword ptr [rax + 228]
    vmovss xmm4, dword ptr [rsi + 40]
    vfmadd231ss xmm0, xmm4, dword ptr [rax + 40]
    vfmadd231ss xmm1, xmm4, dword ptr [rax + 104]
    vfmadd231ss xmm2, xmm4, dword ptr [rax + 168]
    vfmadd231ss xmm3, xmm4, dword ptr [rax + 232]
    vmovss xmm4, dword ptr [rsi + 44]
    vfmadd231ss xmm0, xmm4, dword ptr [rax + 44]
    vfmadd231ss xmm1, xmm4, dword ptr [rax + 108]
    vfmadd231ss xmm2, xmm4, dword ptr [rax + 172]
    vfmadd231ss xmm3, xmm4, dword ptr [rax + 236]
    vmovss xmm4, dword ptr [rsi + 48]
    vfmadd231ss xmm0, xmm4, dword ptr [rax + 48]
    vfmadd231ss xmm1, xmm4, dword ptr [rax + 112]
    vfmadd231ss xmm2, xmm4, dword ptr [rax + 176]
    vfmadd231ss xmm3, xmm4, dword ptr [rax + 240]
    vmovss xmm4, dword ptr [rsi + 52]
    vfmadd231ss xmm0, xmm4, dword ptr [rax + 52]
    vfmadd231ss xmm1, xmm4, dword ptr [rax + 116]
    vfmadd231ss xmm2, xmm4, dword ptr [rax + 180]
    vfmadd231ss xmm3, xmm4, dword ptr [rax + 244]
    vmovss xmm4, dword ptr [rsi + 56]
    vfmadd231ss xmm0, xmm4, dword ptr [rax + 56]
    vfmadd231ss xmm1, xmm4, dword ptr [rax + 120]
    vfmadd231ss xmm2, xmm4, dword ptr [rax + 184]
    vfmadd231ss xmm3, xmm4, dword ptr [rax + 248]
    vmovss xmm4, dword ptr [rsi + 60]
    vfmadd231ss xmm0, xmm4, dword ptr [rax + 60]
    vfmadd231ss xmm1, xmm4, dword ptr [rax + 124]
    vfmadd231ss xmm2, xmm4, dword ptr [rax + 188]
    vfmadd231ss xmm3, xmm4, dword ptr [rax + 252]
    vmovss dword ptr [r8], xmm0
    vmovss dword ptr [r8 + 4], xmm1
    vmovss dword ptr [r8 + 8], xmm2
    vmovss dword ptr [r8 + 12], xmm3
    add rax, 256
    add r8, 16
    dec r9
    jnz 3b
    add rdi, 64
    add rsi, 64
    dec rcx
    jnz 2b
4:
    vzeroupper
    ret
