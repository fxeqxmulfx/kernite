# rmsnorm(gpr, gpr, gpr, gpr) -> ()
# needs: avx, avx2, fma
# frame: 0 bytes aligned to 0, saves []
# y=rdi r=rsi x=rdx n=rcx one=ymm0 ss=ymm1
# v=ymm2,ymm1 s=ymm1
    test rcx, rcx
    jz 3f
    vmovss xmm0, dword ptr [rip + {c0}]
2:
    # the mean square
    vxorps xmm1, xmm1, xmm1
    vmovss xmm2, dword ptr [rdx]
    vfmadd231ss xmm1, xmm2, xmm2
    vmovss xmm2, dword ptr [rdx + 4]
    vfmadd231ss xmm1, xmm2, xmm2
    vmovss xmm2, dword ptr [rdx + 8]
    vfmadd231ss xmm1, xmm2, xmm2
    vmovss xmm2, dword ptr [rdx + 12]
    vfmadd231ss xmm1, xmm2, xmm2
    vmovss xmm2, dword ptr [rdx + 16]
    vfmadd231ss xmm1, xmm2, xmm2
    vmovss xmm2, dword ptr [rdx + 20]
    vfmadd231ss xmm1, xmm2, xmm2
    vmovss xmm2, dword ptr [rdx + 24]
    vfmadd231ss xmm1, xmm2, xmm2
    vmovss xmm2, dword ptr [rdx + 28]
    vfmadd231ss xmm1, xmm2, xmm2
    vmovss xmm2, dword ptr [rdx + 32]
    vfmadd231ss xmm1, xmm2, xmm2
    vmovss xmm2, dword ptr [rdx + 36]
    vfmadd231ss xmm1, xmm2, xmm2
    vmovss xmm2, dword ptr [rdx + 40]
    vfmadd231ss xmm1, xmm2, xmm2
    vmovss xmm2, dword ptr [rdx + 44]
    vfmadd231ss xmm1, xmm2, xmm2
    vmovss xmm2, dword ptr [rdx + 48]
    vfmadd231ss xmm1, xmm2, xmm2
    vmovss xmm2, dword ptr [rdx + 52]
    vfmadd231ss xmm1, xmm2, xmm2
    vmovss xmm2, dword ptr [rdx + 56]
    vfmadd231ss xmm1, xmm2, xmm2
    vmovss xmm2, dword ptr [rdx + 60]
    vfmadd231ss xmm1, xmm2, xmm2
    vmulss xmm1, xmm1, dword ptr [rip + {c1}]
    # its root's reciprocal, and the row by it
    vaddss xmm1, xmm1, dword ptr [rip + {c2}]
    vsqrtss xmm1, xmm1, xmm1
    vdivss xmm1, xmm0, xmm1
    vmovss dword ptr [rsi], xmm1
    vbroadcastss ymm1, xmm1
    vmulps ymm2, ymm1, ymmword ptr [rdx]
    vmovups ymmword ptr [rdi], ymm2
    vmulps ymm1, ymm1, ymmword ptr [rdx + 32]
    vmovups ymmword ptr [rdi + 32], ymm1
    add rdx, 64
    add rdi, 64
    add rsi, 4
    dec rcx
    jnz 2b
3:
    vzeroupper
    ret
# c0, aligned to 4:
#   00 00 80 3f
# c1, aligned to 4:
#   00 00 80 3d
# c2, aligned to 4:
#   ac c5 27 37
