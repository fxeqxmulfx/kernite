# softmax(gpr, gpr, vec, gpr) -> ()
# needs: avx, avx2, fma
# frame: 0 bytes aligned to 0, saves []
# probs=rdi logits=rsi inv_temp=ymm0 n=rdx one=ymm1 p=ymm2,ymm3,ymm5,ymm4
# m=ymm2 k=ymm4,ymm2 total=ymm2 inv=ymm2
    test rdx, rdx
    jz 3f
    vmovss xmm1, dword ptr [rip + {c0}]
    vbroadcastss ymm0, xmm0
2:
    # scaled, and the largest
    vmulps ymm2, ymm0, ymmword ptr [rsi]
    vmovups ymmword ptr [rdi], ymm2
    vmulps ymm2, ymm0, ymmword ptr [rsi + 32]
    vmovups ymmword ptr [rdi + 32], ymm2
    vmulps ymm2, ymm0, ymmword ptr [rsi + 64]
    vmovups ymmword ptr [rdi + 64], ymm2
    vmulps ymm2, ymm0, ymmword ptr [rsi + 96]
    vmovups ymmword ptr [rdi + 96], ymm2
    vmovss xmm2, dword ptr [rip + {c1}]
    vmovss xmm3, dword ptr [rdi]
    vmaxss xmm2, xmm3, xmm2
    vmovss xmm3, dword ptr [rdi + 4]
    vmaxss xmm2, xmm3, xmm2
    vmovss xmm3, dword ptr [rdi + 8]
    vmaxss xmm2, xmm3, xmm2
    vmovss xmm3, dword ptr [rdi + 12]
    vmaxss xmm2, xmm3, xmm2
    vmovss xmm3, dword ptr [rdi + 16]
    vmaxss xmm2, xmm3, xmm2
    vmovss xmm3, dword ptr [rdi + 20]
    vmaxss xmm2, xmm3, xmm2
    vmovss xmm3, dword ptr [rdi + 24]
    vmaxss xmm2, xmm3, xmm2
    vmovss xmm3, dword ptr [rdi + 28]
    vmaxss xmm2, xmm3, xmm2
    vmovss xmm3, dword ptr [rdi + 32]
    vmaxss xmm2, xmm3, xmm2
    vmovss xmm3, dword ptr [rdi + 36]
    vmaxss xmm2, xmm3, xmm2
    vmovss xmm3, dword ptr [rdi + 40]
    vmaxss xmm2, xmm3, xmm2
    vmovss xmm3, dword ptr [rdi + 44]
    vmaxss xmm2, xmm3, xmm2
    vmovss xmm3, dword ptr [rdi + 48]
    vmaxss xmm2, xmm3, xmm2
    vmovss xmm3, dword ptr [rdi + 52]
    vmaxss xmm2, xmm3, xmm2
    vmovss xmm3, dword ptr [rdi + 56]
    vmaxss xmm2, xmm3, xmm2
    vmovss xmm3, dword ptr [rdi + 60]
    vmaxss xmm2, xmm3, xmm2
    vmovss xmm3, dword ptr [rdi + 64]
    vmaxss xmm2, xmm3, xmm2
    vmovss xmm3, dword ptr [rdi + 68]
    vmaxss xmm2, xmm3, xmm2
    vmovss xmm3, dword ptr [rdi + 72]
    vmaxss xmm2, xmm3, xmm2
    vmovss xmm3, dword ptr [rdi + 76]
    vmaxss xmm2, xmm3, xmm2
    vmovss xmm3, dword ptr [rdi + 80]
    vmaxss xmm2, xmm3, xmm2
    vmovss xmm3, dword ptr [rdi + 84]
    vmaxss xmm2, xmm3, xmm2
    vmovss xmm3, dword ptr [rdi + 88]
    vmaxss xmm2, xmm3, xmm2
    vmovss xmm3, dword ptr [rdi + 92]
    vmaxss xmm2, xmm3, xmm2
    vmovss xmm3, dword ptr [rdi + 96]
    vmaxss xmm2, xmm3, xmm2
    vmovss xmm3, dword ptr [rdi + 100]
    vmaxss xmm2, xmm3, xmm2
    vmovss xmm3, dword ptr [rdi + 104]
    vmaxss xmm2, xmm3, xmm2
    # the exponentials, and their sum
    vbroadcastss ymm2, xmm2
    vmovups ymm3, ymmword ptr [rdi]
    vsubps ymm3, ymm3, ymm2
    vmaxps ymm3, ymm3, ymmword ptr [rip + {c2}]
    vmulps ymm4, ymm3, ymmword ptr [rip + {c3}]
    vroundps ymm4, ymm4, 0x08
    vfnmadd231ps ymm3, ymm4, ymmword ptr [rip + {c4}]
    vfnmadd231ps ymm3, ymm4, ymmword ptr [rip + {c5}]
    vmovaps ymm5, ymmword ptr [rip + {c6}]
    vfmadd213ps ymm5, ymm3, ymmword ptr [rip + {c7}]
    vfmadd213ps ymm5, ymm3, ymmword ptr [rip + {c8}]
    vfmadd213ps ymm5, ymm3, ymmword ptr [rip + {c9}]
    vfmadd213ps ymm5, ymm3, ymmword ptr [rip + {c10}]
    vfmadd213ps ymm5, ymm3, ymmword ptr [rip + {c11}]
    vfmadd213ps ymm5, ymm3, ymmword ptr [rip + {c11}]
    vcvtps2dq ymm4, ymm4
    vpaddd ymm4, ymm4, ymmword ptr [rip + {c12}]
    vpslld ymm4, ymm4, 23
    vmulps ymm3, ymm5, ymm4
    vmovups ymmword ptr [rdi], ymm3
    vmovups ymm3, ymmword ptr [rdi + 32]
    vsubps ymm3, ymm3, ymm2
    vmaxps ymm3, ymm3, ymmword ptr [rip + {c2}]
    vmulps ymm4, ymm3, ymmword ptr [rip + {c3}]
    vroundps ymm4, ymm4, 0x08
    vfnmadd231ps ymm3, ymm4, ymmword ptr [rip + {c4}]
    vfnmadd231ps ymm3, ymm4, ymmword ptr [rip + {c5}]
    vmovaps ymm5, ymmword ptr [rip + {c6}]
    vfmadd213ps ymm5, ymm3, ymmword ptr [rip + {c7}]
    vfmadd213ps ymm5, ymm3, ymmword ptr [rip + {c8}]
    vfmadd213ps ymm5, ymm3, ymmword ptr [rip + {c9}]
    vfmadd213ps ymm5, ymm3, ymmword ptr [rip + {c10}]
    vfmadd213ps ymm5, ymm3, ymmword ptr [rip + {c11}]
    vfmadd213ps ymm5, ymm3, ymmword ptr [rip + {c11}]
    vcvtps2dq ymm4, ymm4
    vpaddd ymm4, ymm4, ymmword ptr [rip + {c12}]
    vpslld ymm4, ymm4, 23
    vmulps ymm3, ymm5, ymm4
    vmovups ymmword ptr [rdi + 32], ymm3
    vmovups ymm3, ymmword ptr [rdi + 64]
    vsubps ymm3, ymm3, ymm2
    vmaxps ymm3, ymm3, ymmword ptr [rip + {c2}]
    vmulps ymm4, ymm3, ymmword ptr [rip + {c3}]
    vroundps ymm4, ymm4, 0x08
    vfnmadd231ps ymm3, ymm4, ymmword ptr [rip + {c4}]
    vfnmadd231ps ymm3, ymm4, ymmword ptr [rip + {c5}]
    vmovaps ymm5, ymmword ptr [rip + {c6}]
    vfmadd213ps ymm5, ymm3, ymmword ptr [rip + {c7}]
    vfmadd213ps ymm5, ymm3, ymmword ptr [rip + {c8}]
    vfmadd213ps ymm5, ymm3, ymmword ptr [rip + {c9}]
    vfmadd213ps ymm5, ymm3, ymmword ptr [rip + {c10}]
    vfmadd213ps ymm5, ymm3, ymmword ptr [rip + {c11}]
    vfmadd213ps ymm5, ymm3, ymmword ptr [rip + {c11}]
    vcvtps2dq ymm4, ymm4
    vpaddd ymm4, ymm4, ymmword ptr [rip + {c12}]
    vpslld ymm4, ymm4, 23
    vmulps ymm3, ymm5, ymm4
    vmovups ymmword ptr [rdi + 64], ymm3
    vmovups ymm3, ymmword ptr [rdi + 96]
    vsubps ymm3, ymm3, ymm2
    vmaxps ymm3, ymm3, ymmword ptr [rip + {c2}]
    vmulps ymm2, ymm3, ymmword ptr [rip + {c3}]
    vroundps ymm2, ymm2, 0x08
    vfnmadd231ps ymm3, ymm2, ymmword ptr [rip + {c4}]
    vfnmadd231ps ymm3, ymm2, ymmword ptr [rip + {c5}]
    vmovaps ymm4, ymmword ptr [rip + {c6}]
    vfmadd213ps ymm4, ymm3, ymmword ptr [rip + {c7}]
    vfmadd213ps ymm4, ymm3, ymmword ptr [rip + {c8}]
    vfmadd213ps ymm4, ymm3, ymmword ptr [rip + {c9}]
    vfmadd213ps ymm4, ymm3, ymmword ptr [rip + {c10}]
    vfmadd213ps ymm4, ymm3, ymmword ptr [rip + {c11}]
    vfmadd213ps ymm4, ymm3, ymmword ptr [rip + {c11}]
    vcvtps2dq ymm2, ymm2
    vpaddd ymm2, ymm2, ymmword ptr [rip + {c12}]
    vpslld ymm2, ymm2, 23
    vmulps ymm3, ymm4, ymm2
    vmovups ymmword ptr [rdi + 96], ymm3
    vxorps xmm2, xmm2, xmm2
    vaddss xmm2, xmm2, dword ptr [rdi]
    vaddss xmm2, xmm2, dword ptr [rdi + 4]
    vaddss xmm2, xmm2, dword ptr [rdi + 8]
    vaddss xmm2, xmm2, dword ptr [rdi + 12]
    vaddss xmm2, xmm2, dword ptr [rdi + 16]
    vaddss xmm2, xmm2, dword ptr [rdi + 20]
    vaddss xmm2, xmm2, dword ptr [rdi + 24]
    vaddss xmm2, xmm2, dword ptr [rdi + 28]
    vaddss xmm2, xmm2, dword ptr [rdi + 32]
    vaddss xmm2, xmm2, dword ptr [rdi + 36]
    vaddss xmm2, xmm2, dword ptr [rdi + 40]
    vaddss xmm2, xmm2, dword ptr [rdi + 44]
    vaddss xmm2, xmm2, dword ptr [rdi + 48]
    vaddss xmm2, xmm2, dword ptr [rdi + 52]
    vaddss xmm2, xmm2, dword ptr [rdi + 56]
    vaddss xmm2, xmm2, dword ptr [rdi + 60]
    vaddss xmm2, xmm2, dword ptr [rdi + 64]
    vaddss xmm2, xmm2, dword ptr [rdi + 68]
    vaddss xmm2, xmm2, dword ptr [rdi + 72]
    vaddss xmm2, xmm2, dword ptr [rdi + 76]
    vaddss xmm2, xmm2, dword ptr [rdi + 80]
    vaddss xmm2, xmm2, dword ptr [rdi + 84]
    vaddss xmm2, xmm2, dword ptr [rdi + 88]
    vaddss xmm2, xmm2, dword ptr [rdi + 92]
    vaddss xmm2, xmm2, dword ptr [rdi + 96]
    vaddss xmm2, xmm2, dword ptr [rdi + 100]
    vaddss xmm2, xmm2, dword ptr [rdi + 104]
    # each divided by it, past V zero
    vdivss xmm2, xmm1, xmm2
    vbroadcastss ymm2, xmm2
    vmulps ymm3, ymm2, ymmword ptr [rdi]
    vmovups ymmword ptr [rdi], ymm3
    vmulps ymm3, ymm2, ymmword ptr [rdi + 32]
    vmovups ymmword ptr [rdi + 32], ymm3
    vmulps ymm3, ymm2, ymmword ptr [rdi + 64]
    vmovups ymmword ptr [rdi + 64], ymm3
    vmulps ymm2, ymm2, ymmword ptr [rdi + 96]
    vandps ymm2, ymm2, ymmword ptr [rip + {c13}]
    vmovups ymmword ptr [rdi + 96], ymm2
    add rdi, 128
    add rsi, 128
    dec rdx
    jnz 2b
3:
    vzeroupper
    ret
# c0, aligned to 4:
#   00 00 80 3f
# c1, aligned to 4:
#   00 00 80 ff
# c2, aligned to 32:
#   00 00 ae c2 00 00 ae c2 00 00 ae c2 00 00 ae c2
#   00 00 ae c2 00 00 ae c2 00 00 ae c2 00 00 ae c2
# c3, aligned to 32:
#   3b aa b8 3f 3b aa b8 3f 3b aa b8 3f 3b aa b8 3f
#   3b aa b8 3f 3b aa b8 3f 3b aa b8 3f 3b aa b8 3f
# c4, aligned to 32:
#   00 80 31 3f 00 80 31 3f 00 80 31 3f 00 80 31 3f
#   00 80 31 3f 00 80 31 3f 00 80 31 3f 00 80 31 3f
# c5, aligned to 32:
#   83 80 5e b9 83 80 5e b9 83 80 5e b9 83 80 5e b9
#   83 80 5e b9 83 80 5e b9 83 80 5e b9 83 80 5e b9
# c6, aligned to 32:
#   61 0b b6 3a 61 0b b6 3a 61 0b b6 3a 61 0b b6 3a
#   61 0b b6 3a 61 0b b6 3a 61 0b b6 3a 61 0b b6 3a
# c7, aligned to 32:
#   89 88 08 3c 89 88 08 3c 89 88 08 3c 89 88 08 3c
#   89 88 08 3c 89 88 08 3c 89 88 08 3c 89 88 08 3c
# c8, aligned to 32:
#   ab aa 2a 3d ab aa 2a 3d ab aa 2a 3d ab aa 2a 3d
#   ab aa 2a 3d ab aa 2a 3d ab aa 2a 3d ab aa 2a 3d
# c9, aligned to 32:
#   ab aa 2a 3e ab aa 2a 3e ab aa 2a 3e ab aa 2a 3e
#   ab aa 2a 3e ab aa 2a 3e ab aa 2a 3e ab aa 2a 3e
# c10, aligned to 32:
#   00 00 00 3f 00 00 00 3f 00 00 00 3f 00 00 00 3f
#   00 00 00 3f 00 00 00 3f 00 00 00 3f 00 00 00 3f
# c11, aligned to 32:
#   00 00 80 3f 00 00 80 3f 00 00 80 3f 00 00 80 3f
#   00 00 80 3f 00 00 80 3f 00 00 80 3f 00 00 80 3f
# c12, aligned to 32:
#   7f 00 00 00 7f 00 00 00 7f 00 00 00 7f 00 00 00
#   7f 00 00 00 7f 00 00 00 7f 00 00 00 7f 00 00 00
# c13, aligned to 32:
#   ff ff ff ff ff ff ff ff ff ff ff ff 00 00 00 00
#   00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
