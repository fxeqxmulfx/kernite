# attention(gpr, gpr, gpr, gpr, gpr) -> ()
# needs: avx, avx2, fma
# frame: 0 bytes aligned to 0, saves [rbx, r12]
# att=rdi aw=rsi qkv=rdx t0=rcx n=r8 one=ymm0
# q=rax w=r9 out=r10 qj0=ymm1 qj1=ymm2 qj2=ymm3
# qj3=ymm4 m=ymm5 ks=r11,rax ws=rbx,r11 left=r12,rbx l=ymm6
# total=ymm1 x=ymm2 k=ymm3 p=ymm4 inv=ymm1 sum=ymm2
# ww=ymm3
    push rbx
    push r12
    cmp rcx, r8
    jae 17f
    vmovss xmm0, dword ptr [rip + {c0}]
2:
    imul rax, rcx, 192
    add rax, rdx
    imul r9, rcx, 256
    add r9, rsi
    imul r10, rcx, 64
    add r10, rdi
    # head 0: the logits, and the largest
    vmovss xmm1, dword ptr [rax]
    vmovss xmm2, dword ptr [rax + 4]
    vmovss xmm3, dword ptr [rax + 8]
    vmovss xmm4, dword ptr [rax + 12]
    vmovss xmm5, dword ptr [rip + {c1}]
    lea r11, [rdx + 64]
    lea rbx, [r9]
    lea r12, [rcx + 1]
3:
    vxorps xmm6, xmm6, xmm6
    vfmadd231ss xmm6, xmm1, dword ptr [r11]
    vfmadd231ss xmm6, xmm2, dword ptr [r11 + 4]
    vfmadd231ss xmm6, xmm3, dword ptr [r11 + 8]
    vfmadd231ss xmm6, xmm4, dword ptr [r11 + 12]
    vmulss xmm6, xmm6, dword ptr [rip + {c2}]
    vmovss dword ptr [rbx], xmm6
    vmaxss xmm5, xmm6, xmm5
    add r11, 192
    add rbx, 4
    dec r12
    jnz 3b
    # head 0: their exponentials, and the sum
    vxorps xmm1, xmm1, xmm1
    lea rbx, [r9]
    lea r12, [rcx + 1]
4:
    vmovss xmm2, dword ptr [rbx]
    vsubss xmm2, xmm2, xmm5
    vmaxps ymm2, ymm2, ymmword ptr [rip + {c3}]
    vmulps ymm3, ymm2, ymmword ptr [rip + {c4}]
    vroundps ymm3, ymm3, 0x08
    vfnmadd231ps ymm2, ymm3, ymmword ptr [rip + {c5}]
    vfnmadd231ps ymm2, ymm3, ymmword ptr [rip + {c6}]
    vmovaps ymm4, ymmword ptr [rip + {c7}]
    vfmadd213ps ymm4, ymm2, ymmword ptr [rip + {c8}]
    vfmadd213ps ymm4, ymm2, ymmword ptr [rip + {c9}]
    vfmadd213ps ymm4, ymm2, ymmword ptr [rip + {c10}]
    vfmadd213ps ymm4, ymm2, ymmword ptr [rip + {c11}]
    vfmadd213ps ymm4, ymm2, ymmword ptr [rip + {c12}]
    vfmadd213ps ymm4, ymm2, ymmword ptr [rip + {c12}]
    vcvtps2dq ymm3, ymm3
    vpaddd ymm3, ymm3, ymmword ptr [rip + {c13}]
    vpslld ymm3, ymm3, 23
    vmulps ymm2, ymm4, ymm3
    vmovss dword ptr [rbx], xmm2
    vaddss xmm1, xmm1, xmm2
    add rbx, 4
    dec r12
    jnz 4b
    # head 0: the weights, and the values they weigh
    vdivss xmm1, xmm0, xmm1
    vxorps xmm2, xmm2, xmm2
    lea r11, [rdx + 128]
    lea rbx, [r9]
    lea r12, [rcx + 1]
5:
    vmulss xmm3, xmm1, dword ptr [rbx]
    vmovss dword ptr [rbx], xmm3
    vbroadcastss xmm3, xmm3
    vfmadd231ps xmm2, xmm3, xmmword ptr [r11]
    add r11, 192
    add rbx, 4
    dec r12
    jnz 5b
    vmovups xmmword ptr [r10], xmm2
    # head 1: the logits, and the largest
    vmovss xmm1, dword ptr [rax + 16]
    vmovss xmm2, dword ptr [rax + 20]
    vmovss xmm3, dword ptr [rax + 24]
    vmovss xmm4, dword ptr [rax + 28]
    vmovss xmm5, dword ptr [rip + {c1}]
    lea r11, [rdx + 80]
    lea rbx, [r9 + 64]
    lea r12, [rcx + 1]
6:
    vxorps xmm6, xmm6, xmm6
    vfmadd231ss xmm6, xmm1, dword ptr [r11]
    vfmadd231ss xmm6, xmm2, dword ptr [r11 + 4]
    vfmadd231ss xmm6, xmm3, dword ptr [r11 + 8]
    vfmadd231ss xmm6, xmm4, dword ptr [r11 + 12]
    vmulss xmm6, xmm6, dword ptr [rip + {c2}]
    vmovss dword ptr [rbx], xmm6
    vmaxss xmm5, xmm6, xmm5
    add r11, 192
    add rbx, 4
    dec r12
    jnz 6b
    # head 1: their exponentials, and the sum
    vxorps xmm1, xmm1, xmm1
    lea rbx, [r9 + 64]
    lea r12, [rcx + 1]
7:
    vmovss xmm2, dword ptr [rbx]
    vsubss xmm2, xmm2, xmm5
    vmaxps ymm2, ymm2, ymmword ptr [rip + {c3}]
    vmulps ymm3, ymm2, ymmword ptr [rip + {c4}]
    vroundps ymm3, ymm3, 0x08
    vfnmadd231ps ymm2, ymm3, ymmword ptr [rip + {c5}]
    vfnmadd231ps ymm2, ymm3, ymmword ptr [rip + {c6}]
    vmovaps ymm4, ymmword ptr [rip + {c7}]
    vfmadd213ps ymm4, ymm2, ymmword ptr [rip + {c8}]
    vfmadd213ps ymm4, ymm2, ymmword ptr [rip + {c9}]
    vfmadd213ps ymm4, ymm2, ymmword ptr [rip + {c10}]
    vfmadd213ps ymm4, ymm2, ymmword ptr [rip + {c11}]
    vfmadd213ps ymm4, ymm2, ymmword ptr [rip + {c12}]
    vfmadd213ps ymm4, ymm2, ymmword ptr [rip + {c12}]
    vcvtps2dq ymm3, ymm3
    vpaddd ymm3, ymm3, ymmword ptr [rip + {c13}]
    vpslld ymm3, ymm3, 23
    vmulps ymm2, ymm4, ymm3
    vmovss dword ptr [rbx], xmm2
    vaddss xmm1, xmm1, xmm2
    add rbx, 4
    dec r12
    jnz 7b
    # head 1: the weights, and the values they weigh
    vdivss xmm1, xmm0, xmm1
    vxorps xmm2, xmm2, xmm2
    lea r11, [rdx + 144]
    lea rbx, [r9 + 64]
    lea r12, [rcx + 1]
8:
    vmulss xmm3, xmm1, dword ptr [rbx]
    vmovss dword ptr [rbx], xmm3
    vbroadcastss xmm3, xmm3
    vfmadd231ps xmm2, xmm3, xmmword ptr [r11]
    add r11, 192
    add rbx, 4
    dec r12
    jnz 8b
    vmovups xmmword ptr [r10 + 16], xmm2
    # head 2: the logits, and the largest
    vmovss xmm1, dword ptr [rax + 32]
    vmovss xmm2, dword ptr [rax + 36]
    vmovss xmm3, dword ptr [rax + 40]
    vmovss xmm4, dword ptr [rax + 44]
    vmovss xmm5, dword ptr [rip + {c1}]
    lea r11, [rdx + 96]
    lea rbx, [r9 + 128]
    lea r12, [rcx + 1]
9:
    vxorps xmm6, xmm6, xmm6
    vfmadd231ss xmm6, xmm1, dword ptr [r11]
    vfmadd231ss xmm6, xmm2, dword ptr [r11 + 4]
    vfmadd231ss xmm6, xmm3, dword ptr [r11 + 8]
    vfmadd231ss xmm6, xmm4, dword ptr [r11 + 12]
    vmulss xmm6, xmm6, dword ptr [rip + {c2}]
    vmovss dword ptr [rbx], xmm6
    vmaxss xmm5, xmm6, xmm5
    add r11, 192
    add rbx, 4
    dec r12
    jnz 9b
    # head 2: their exponentials, and the sum
    vxorps xmm1, xmm1, xmm1
    lea rbx, [r9 + 128]
    lea r12, [rcx + 1]
12:
    vmovss xmm2, dword ptr [rbx]
    vsubss xmm2, xmm2, xmm5
    vmaxps ymm2, ymm2, ymmword ptr [rip + {c3}]
    vmulps ymm3, ymm2, ymmword ptr [rip + {c4}]
    vroundps ymm3, ymm3, 0x08
    vfnmadd231ps ymm2, ymm3, ymmword ptr [rip + {c5}]
    vfnmadd231ps ymm2, ymm3, ymmword ptr [rip + {c6}]
    vmovaps ymm4, ymmword ptr [rip + {c7}]
    vfmadd213ps ymm4, ymm2, ymmword ptr [rip + {c8}]
    vfmadd213ps ymm4, ymm2, ymmword ptr [rip + {c9}]
    vfmadd213ps ymm4, ymm2, ymmword ptr [rip + {c10}]
    vfmadd213ps ymm4, ymm2, ymmword ptr [rip + {c11}]
    vfmadd213ps ymm4, ymm2, ymmword ptr [rip + {c12}]
    vfmadd213ps ymm4, ymm2, ymmword ptr [rip + {c12}]
    vcvtps2dq ymm3, ymm3
    vpaddd ymm3, ymm3, ymmword ptr [rip + {c13}]
    vpslld ymm3, ymm3, 23
    vmulps ymm2, ymm4, ymm3
    vmovss dword ptr [rbx], xmm2
    vaddss xmm1, xmm1, xmm2
    add rbx, 4
    dec r12
    jnz 12b
    # head 2: the weights, and the values they weigh
    vdivss xmm1, xmm0, xmm1
    vxorps xmm2, xmm2, xmm2
    lea r11, [rdx + 160]
    lea rbx, [r9 + 128]
    lea r12, [rcx + 1]
13:
    vmulss xmm3, xmm1, dword ptr [rbx]
    vmovss dword ptr [rbx], xmm3
    vbroadcastss xmm3, xmm3
    vfmadd231ps xmm2, xmm3, xmmword ptr [r11]
    add r11, 192
    add rbx, 4
    dec r12
    jnz 13b
    vmovups xmmword ptr [r10 + 32], xmm2
    # head 3: the logits, and the largest
    vmovss xmm1, dword ptr [rax + 48]
    vmovss xmm2, dword ptr [rax + 52]
    vmovss xmm3, dword ptr [rax + 56]
    vmovss xmm4, dword ptr [rax + 60]
    vmovss xmm5, dword ptr [rip + {c1}]
    lea rax, [rdx + 112]
    lea r11, [r9 + 192]
    lea rbx, [rcx + 1]
14:
    vxorps xmm6, xmm6, xmm6
    vfmadd231ss xmm6, xmm1, dword ptr [rax]
    vfmadd231ss xmm6, xmm2, dword ptr [rax + 4]
    vfmadd231ss xmm6, xmm3, dword ptr [rax + 8]
    vfmadd231ss xmm6, xmm4, dword ptr [rax + 12]
    vmulss xmm6, xmm6, dword ptr [rip + {c2}]
    vmovss dword ptr [r11], xmm6
    vmaxss xmm5, xmm6, xmm5
    add rax, 192
    add r11, 4
    dec rbx
    jnz 14b
    # head 3: their exponentials, and the sum
    vxorps xmm1, xmm1, xmm1
    lea r11, [r9 + 192]
    lea rbx, [rcx + 1]
15:
    vmovss xmm2, dword ptr [r11]
    vsubss xmm2, xmm2, xmm5
    vmaxps ymm2, ymm2, ymmword ptr [rip + {c3}]
    vmulps ymm3, ymm2, ymmword ptr [rip + {c4}]
    vroundps ymm3, ymm3, 0x08
    vfnmadd231ps ymm2, ymm3, ymmword ptr [rip + {c5}]
    vfnmadd231ps ymm2, ymm3, ymmword ptr [rip + {c6}]
    vmovaps ymm4, ymmword ptr [rip + {c7}]
    vfmadd213ps ymm4, ymm2, ymmword ptr [rip + {c8}]
    vfmadd213ps ymm4, ymm2, ymmword ptr [rip + {c9}]
    vfmadd213ps ymm4, ymm2, ymmword ptr [rip + {c10}]
    vfmadd213ps ymm4, ymm2, ymmword ptr [rip + {c11}]
    vfmadd213ps ymm4, ymm2, ymmword ptr [rip + {c12}]
    vfmadd213ps ymm4, ymm2, ymmword ptr [rip + {c12}]
    vcvtps2dq ymm3, ymm3
    vpaddd ymm3, ymm3, ymmword ptr [rip + {c13}]
    vpslld ymm3, ymm3, 23
    vmulps ymm2, ymm4, ymm3
    vmovss dword ptr [r11], xmm2
    vaddss xmm1, xmm1, xmm2
    add r11, 4
    dec rbx
    jnz 15b
    # head 3: the weights, and the values they weigh
    vdivss xmm1, xmm0, xmm1
    vxorps xmm2, xmm2, xmm2
    lea rax, [rdx + 176]
    lea r11, [r9 + 192]
    lea rbx, [rcx + 1]
16:
    vmulss xmm3, xmm1, dword ptr [r11]
    vmovss dword ptr [r11], xmm3
    vbroadcastss xmm3, xmm3
    vfmadd231ps xmm2, xmm3, xmmword ptr [rax]
    add rax, 192
    add r11, 4
    dec rbx
    jnz 16b
    vmovups xmmword ptr [r10 + 48], xmm2
    inc rcx
    cmp rcx, r8
    jb 2b
17:
    pop r12
    pop rbx
    vzeroupper
    ret
# c0, aligned to 4:
#   00 00 80 3f
# c1, aligned to 4:
#   00 00 80 ff
# c2, aligned to 4:
#   00 00 00 3f
# c3, aligned to 32:
#   00 00 ae c2 00 00 ae c2 00 00 ae c2 00 00 ae c2
#   00 00 ae c2 00 00 ae c2 00 00 ae c2 00 00 ae c2
# c4, aligned to 32:
#   3b aa b8 3f 3b aa b8 3f 3b aa b8 3f 3b aa b8 3f
#   3b aa b8 3f 3b aa b8 3f 3b aa b8 3f 3b aa b8 3f
# c5, aligned to 32:
#   00 80 31 3f 00 80 31 3f 00 80 31 3f 00 80 31 3f
#   00 80 31 3f 00 80 31 3f 00 80 31 3f 00 80 31 3f
# c6, aligned to 32:
#   83 80 5e b9 83 80 5e b9 83 80 5e b9 83 80 5e b9
#   83 80 5e b9 83 80 5e b9 83 80 5e b9 83 80 5e b9
# c7, aligned to 32:
#   61 0b b6 3a 61 0b b6 3a 61 0b b6 3a 61 0b b6 3a
#   61 0b b6 3a 61 0b b6 3a 61 0b b6 3a 61 0b b6 3a
# c8, aligned to 32:
#   89 88 08 3c 89 88 08 3c 89 88 08 3c 89 88 08 3c
#   89 88 08 3c 89 88 08 3c 89 88 08 3c 89 88 08 3c
# c9, aligned to 32:
#   ab aa 2a 3d ab aa 2a 3d ab aa 2a 3d ab aa 2a 3d
#   ab aa 2a 3d ab aa 2a 3d ab aa 2a 3d ab aa 2a 3d
# c10, aligned to 32:
#   ab aa 2a 3e ab aa 2a 3e ab aa 2a 3e ab aa 2a 3e
#   ab aa 2a 3e ab aa 2a 3e ab aa 2a 3e ab aa 2a 3e
# c11, aligned to 32:
#   00 00 00 3f 00 00 00 3f 00 00 00 3f 00 00 00 3f
#   00 00 00 3f 00 00 00 3f 00 00 00 3f 00 00 00 3f
# c12, aligned to 32:
#   00 00 80 3f 00 00 80 3f 00 00 80 3f 00 00 80 3f
#   00 00 80 3f 00 00 80 3f 00 00 80 3f 00 00 80 3f
# c13, aligned to 32:
#   7f 00 00 00 7f 00 00 00 7f 00 00 00 7f 00 00 00
#   7f 00 00 00 7f 00 00 00 7f 00 00 00 7f 00 00 00
