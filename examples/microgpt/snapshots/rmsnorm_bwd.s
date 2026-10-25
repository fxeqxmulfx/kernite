# rmsnorm_bwd(gpr, gpr, gpr, gpr, gpr) -> ()
# frame: 0 bytes aligned to 0, saves []
# dx=rdi dy=rsi x=rdx r=rcx n=r8 dot=ymm0
# v=ymm1 s=ymm1 c=ymm2 g=ymm0
    test r8, r8
    jz 3f
2:
    vxorps xmm0, xmm0, xmm0
    vmovss xmm1, dword ptr [rsi]
    vfmadd231ss xmm0, xmm1, dword ptr [rdx]
    vmovss xmm1, dword ptr [rsi + 4]
    vfmadd231ss xmm0, xmm1, dword ptr [rdx + 4]
    vmovss xmm1, dword ptr [rsi + 8]
    vfmadd231ss xmm0, xmm1, dword ptr [rdx + 8]
    vmovss xmm1, dword ptr [rsi + 12]
    vfmadd231ss xmm0, xmm1, dword ptr [rdx + 12]
    vmovss xmm1, dword ptr [rsi + 16]
    vfmadd231ss xmm0, xmm1, dword ptr [rdx + 16]
    vmovss xmm1, dword ptr [rsi + 20]
    vfmadd231ss xmm0, xmm1, dword ptr [rdx + 20]
    vmovss xmm1, dword ptr [rsi + 24]
    vfmadd231ss xmm0, xmm1, dword ptr [rdx + 24]
    vmovss xmm1, dword ptr [rsi + 28]
    vfmadd231ss xmm0, xmm1, dword ptr [rdx + 28]
    vmovss xmm1, dword ptr [rsi + 32]
    vfmadd231ss xmm0, xmm1, dword ptr [rdx + 32]
    vmovss xmm1, dword ptr [rsi + 36]
    vfmadd231ss xmm0, xmm1, dword ptr [rdx + 36]
    vmovss xmm1, dword ptr [rsi + 40]
    vfmadd231ss xmm0, xmm1, dword ptr [rdx + 40]
    vmovss xmm1, dword ptr [rsi + 44]
    vfmadd231ss xmm0, xmm1, dword ptr [rdx + 44]
    vmovss xmm1, dword ptr [rsi + 48]
    vfmadd231ss xmm0, xmm1, dword ptr [rdx + 48]
    vmovss xmm1, dword ptr [rsi + 52]
    vfmadd231ss xmm0, xmm1, dword ptr [rdx + 52]
    vmovss xmm1, dword ptr [rsi + 56]
    vfmadd231ss xmm0, xmm1, dword ptr [rdx + 56]
    vmovss xmm1, dword ptr [rsi + 60]
    vfmadd231ss xmm0, xmm1, dword ptr [rdx + 60]
    # c = dot r^3 / E
    vmovss xmm1, dword ptr [rcx]
    vmulss xmm2, xmm1, xmm1
    vmulss xmm2, xmm2, xmm1
    vmulss xmm2, xmm0, xmm2
    vmulss xmm2, xmm2, dword ptr [rip + {c0}]
    vbroadcastss ymm2, xmm2
    vbroadcastss ymm1, xmm1
    # dx += dy r - c x
    vmulps ymm0, ymm1, ymmword ptr [rsi]
    vfnmadd231ps ymm0, ymm2, ymmword ptr [rdx]
    vaddps ymm0, ymm0, ymmword ptr [rdi]
    vmovups ymmword ptr [rdi], ymm0
    vmulps ymm0, ymm1, ymmword ptr [rsi + 32]
    vfnmadd231ps ymm0, ymm2, ymmword ptr [rdx + 32]
    vaddps ymm0, ymm0, ymmword ptr [rdi + 32]
    vmovups ymmword ptr [rdi + 32], ymm0
    add rdi, 64
    add rsi, 64
    add rdx, 64
    add rcx, 4
    dec r8
    jnz 2b
3:
    vzeroupper
    ret
# c0, aligned to 4:
#   00 00 80 3d
