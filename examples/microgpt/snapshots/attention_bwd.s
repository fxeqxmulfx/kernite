# attention_bwd(gpr, gpr, gpr, gpr, gpr) -> ()
# needs: avx, avx2, fma
# frame: 0 bytes aligned to 0, saves [rbx, r12, r13, r14]
# dqkv=rdi qkv=rsi aw=rdx datt=rcx n=r8 t=rax
# w=r9 at=r10 dout0=ymm0 dout1=ymm1 dout2=ymm2 dout3=ymm3
# douts=ymm4 vs=r10 ws=r11 left=rbx sum=ymm5 dw=ymm6,ymm8
# ww=ymm7,ymm9 q=r12 ks=r13 dks=r14 qs=ymm6 dq=ymm7
# acc=ymm10
    push rbx
    push r12
    push r13
    push r14
    test r8, r8
    jz 16f
    xor eax, eax
2:
    # head 0: the output's gradient
    imul r9, rax, 256
    add r9, rdx
    imul r10, rax, 64
    vmovss xmm0, dword ptr [rcx + r10*1]
    vmovss xmm1, dword ptr [rcx + r10*1 + 4]
    vmovss xmm2, dword ptr [rcx + r10*1 + 8]
    vmovss xmm3, dword ptr [rcx + r10*1 + 12]
    vmovups xmm4, xmmword ptr [rcx + r10*1]
    # head 0: the sum of the weights by their gradients
    lea r10, [rsi + 128]
    lea r11, [r9]
    lea rbx, [rax + 1]
    vxorps xmm5, xmm5, xmm5
3:
    vxorps xmm6, xmm6, xmm6
    vfmadd231ss xmm6, xmm0, dword ptr [r10]
    vfmadd231ss xmm6, xmm1, dword ptr [r10 + 4]
    vfmadd231ss xmm6, xmm2, dword ptr [r10 + 8]
    vfmadd231ss xmm6, xmm3, dword ptr [r10 + 12]
    vmovss xmm7, dword ptr [r11]
    vfmadd231ss xmm5, xmm7, xmm6
    add r10, 192
    add r11, 4
    dec rbx
    jnz 3b
    # head 0: the query's gradient, and the keys' and values'
    imul r12, rax, 192
    vmovups xmm6, xmmword ptr [rsi + r12*1]
    lea r13, [rsi + 64]
    lea r14, [rdi + 64]
    lea r10, [rsi + 128]
    lea r11, [r9]
    lea rbx, [rax + 1]
    vxorps xmm7, xmm7, xmm7
4:
    vxorps xmm8, xmm8, xmm8
    vfmadd231ss xmm8, xmm0, dword ptr [r10]
    vfmadd231ss xmm8, xmm1, dword ptr [r10 + 4]
    vfmadd231ss xmm8, xmm2, dword ptr [r10 + 8]
    vfmadd231ss xmm8, xmm3, dword ptr [r10 + 12]
    vsubss xmm8, xmm8, xmm5
    vmovss xmm9, dword ptr [r11]
    vmulss xmm8, xmm9, xmm8
    vmulss xmm8, xmm8, dword ptr [rip + {c0}]
    vbroadcastss xmm8, xmm8
    vfmadd231ps xmm7, xmm8, xmmword ptr [r13]
    vmovups xmm10, xmmword ptr [r14]
    vfmadd231ps xmm10, xmm8, xmm6
    vmovups xmmword ptr [r14], xmm10
    vbroadcastss xmm9, xmm9
    vmovups xmm10, xmmword ptr [r14 + 64]
    vfmadd231ps xmm10, xmm9, xmm4
    vmovups xmmword ptr [r14 + 64], xmm10
    add r13, 192
    add r10, 192
    add r14, 192
    add r11, 4
    dec rbx
    jnz 4b
    vmovups xmmword ptr [rdi + r12*1], xmm7
    inc rax
    cmp rax, r8
    jb 2b
    xor eax, eax
5:
    # head 1: the output's gradient
    imul r9, rax, 256
    add r9, rdx
    imul r10, rax, 64
    vmovss xmm0, dword ptr [rcx + r10*1 + 16]
    vmovss xmm1, dword ptr [rcx + r10*1 + 20]
    vmovss xmm2, dword ptr [rcx + r10*1 + 24]
    vmovss xmm3, dword ptr [rcx + r10*1 + 28]
    vmovups xmm4, xmmword ptr [rcx + r10*1 + 16]
    # head 1: the sum of the weights by their gradients
    lea r10, [rsi + 144]
    lea r11, [r9 + 64]
    lea rbx, [rax + 1]
    vxorps xmm5, xmm5, xmm5
6:
    vxorps xmm6, xmm6, xmm6
    vfmadd231ss xmm6, xmm0, dword ptr [r10]
    vfmadd231ss xmm6, xmm1, dword ptr [r10 + 4]
    vfmadd231ss xmm6, xmm2, dword ptr [r10 + 8]
    vfmadd231ss xmm6, xmm3, dword ptr [r10 + 12]
    vmovss xmm7, dword ptr [r11]
    vfmadd231ss xmm5, xmm7, xmm6
    add r10, 192
    add r11, 4
    dec rbx
    jnz 6b
    # head 1: the query's gradient, and the keys' and values'
    imul r12, rax, 192
    vmovups xmm6, xmmword ptr [rsi + r12*1 + 16]
    lea r13, [rsi + 80]
    lea r14, [rdi + 80]
    lea r10, [rsi + 144]
    lea r11, [r9 + 64]
    lea rbx, [rax + 1]
    vxorps xmm7, xmm7, xmm7
7:
    vxorps xmm8, xmm8, xmm8
    vfmadd231ss xmm8, xmm0, dword ptr [r10]
    vfmadd231ss xmm8, xmm1, dword ptr [r10 + 4]
    vfmadd231ss xmm8, xmm2, dword ptr [r10 + 8]
    vfmadd231ss xmm8, xmm3, dword ptr [r10 + 12]
    vsubss xmm8, xmm8, xmm5
    vmovss xmm9, dword ptr [r11]
    vmulss xmm8, xmm9, xmm8
    vmulss xmm8, xmm8, dword ptr [rip + {c0}]
    vbroadcastss xmm8, xmm8
    vfmadd231ps xmm7, xmm8, xmmword ptr [r13]
    vmovups xmm10, xmmword ptr [r14]
    vfmadd231ps xmm10, xmm8, xmm6
    vmovups xmmword ptr [r14], xmm10
    vbroadcastss xmm9, xmm9
    vmovups xmm10, xmmword ptr [r14 + 64]
    vfmadd231ps xmm10, xmm9, xmm4
    vmovups xmmword ptr [r14 + 64], xmm10
    add r13, 192
    add r10, 192
    add r14, 192
    add r11, 4
    dec rbx
    jnz 7b
    vmovups xmmword ptr [rdi + r12*1 + 16], xmm7
    inc rax
    cmp rax, r8
    jb 5b
    xor eax, eax
8:
    # head 2: the output's gradient
    imul r9, rax, 256
    add r9, rdx
    imul r10, rax, 64
    vmovss xmm0, dword ptr [rcx + r10*1 + 32]
    vmovss xmm1, dword ptr [rcx + r10*1 + 36]
    vmovss xmm2, dword ptr [rcx + r10*1 + 40]
    vmovss xmm3, dword ptr [rcx + r10*1 + 44]
    vmovups xmm4, xmmword ptr [rcx + r10*1 + 32]
    # head 2: the sum of the weights by their gradients
    lea r10, [rsi + 160]
    lea r11, [r9 + 128]
    lea rbx, [rax + 1]
    vxorps xmm5, xmm5, xmm5
9:
    vxorps xmm6, xmm6, xmm6
    vfmadd231ss xmm6, xmm0, dword ptr [r10]
    vfmadd231ss xmm6, xmm1, dword ptr [r10 + 4]
    vfmadd231ss xmm6, xmm2, dword ptr [r10 + 8]
    vfmadd231ss xmm6, xmm3, dword ptr [r10 + 12]
    vmovss xmm7, dword ptr [r11]
    vfmadd231ss xmm5, xmm7, xmm6
    add r10, 192
    add r11, 4
    dec rbx
    jnz 9b
    # head 2: the query's gradient, and the keys' and values'
    imul r12, rax, 192
    vmovups xmm6, xmmword ptr [rsi + r12*1 + 32]
    lea r13, [rsi + 96]
    lea r14, [rdi + 96]
    lea r10, [rsi + 160]
    lea r11, [r9 + 128]
    lea rbx, [rax + 1]
    vxorps xmm7, xmm7, xmm7
12:
    vxorps xmm8, xmm8, xmm8
    vfmadd231ss xmm8, xmm0, dword ptr [r10]
    vfmadd231ss xmm8, xmm1, dword ptr [r10 + 4]
    vfmadd231ss xmm8, xmm2, dword ptr [r10 + 8]
    vfmadd231ss xmm8, xmm3, dword ptr [r10 + 12]
    vsubss xmm8, xmm8, xmm5
    vmovss xmm9, dword ptr [r11]
    vmulss xmm8, xmm9, xmm8
    vmulss xmm8, xmm8, dword ptr [rip + {c0}]
    vbroadcastss xmm8, xmm8
    vfmadd231ps xmm7, xmm8, xmmword ptr [r13]
    vmovups xmm10, xmmword ptr [r14]
    vfmadd231ps xmm10, xmm8, xmm6
    vmovups xmmword ptr [r14], xmm10
    vbroadcastss xmm9, xmm9
    vmovups xmm10, xmmword ptr [r14 + 64]
    vfmadd231ps xmm10, xmm9, xmm4
    vmovups xmmword ptr [r14 + 64], xmm10
    add r13, 192
    add r10, 192
    add r14, 192
    add r11, 4
    dec rbx
    jnz 12b
    vmovups xmmword ptr [rdi + r12*1 + 32], xmm7
    inc rax
    cmp rax, r8
    jb 8b
    xor eax, eax
13:
    # head 3: the output's gradient
    imul r9, rax, 256
    add r9, rdx
    imul r10, rax, 64
    vmovss xmm0, dword ptr [rcx + r10*1 + 48]
    vmovss xmm1, dword ptr [rcx + r10*1 + 52]
    vmovss xmm2, dword ptr [rcx + r10*1 + 56]
    vmovss xmm3, dword ptr [rcx + r10*1 + 60]
    vmovups xmm4, xmmword ptr [rcx + r10*1 + 48]
    # head 3: the sum of the weights by their gradients
    lea r10, [rsi + 176]
    lea r11, [r9 + 192]
    lea rbx, [rax + 1]
    vxorps xmm5, xmm5, xmm5
14:
    vxorps xmm6, xmm6, xmm6
    vfmadd231ss xmm6, xmm0, dword ptr [r10]
    vfmadd231ss xmm6, xmm1, dword ptr [r10 + 4]
    vfmadd231ss xmm6, xmm2, dword ptr [r10 + 8]
    vfmadd231ss xmm6, xmm3, dword ptr [r10 + 12]
    vmovss xmm7, dword ptr [r11]
    vfmadd231ss xmm5, xmm7, xmm6
    add r10, 192
    add r11, 4
    dec rbx
    jnz 14b
    # head 3: the query's gradient, and the keys' and values'
    imul r12, rax, 192
    vmovups xmm6, xmmword ptr [rsi + r12*1 + 48]
    lea r13, [rsi + 112]
    lea r14, [rdi + 112]
    lea r10, [rsi + 176]
    lea r11, [r9 + 192]
    lea rbx, [rax + 1]
    vxorps xmm7, xmm7, xmm7
15:
    vxorps xmm8, xmm8, xmm8
    vfmadd231ss xmm8, xmm0, dword ptr [r10]
    vfmadd231ss xmm8, xmm1, dword ptr [r10 + 4]
    vfmadd231ss xmm8, xmm2, dword ptr [r10 + 8]
    vfmadd231ss xmm8, xmm3, dword ptr [r10 + 12]
    vsubss xmm8, xmm8, xmm5
    vmovss xmm9, dword ptr [r11]
    vmulss xmm8, xmm9, xmm8
    vmulss xmm8, xmm8, dword ptr [rip + {c0}]
    vbroadcastss xmm8, xmm8
    vfmadd231ps xmm7, xmm8, xmmword ptr [r13]
    vmovups xmm10, xmmword ptr [r14]
    vfmadd231ps xmm10, xmm8, xmm6
    vmovups xmmword ptr [r14], xmm10
    vbroadcastss xmm9, xmm9
    vmovups xmm10, xmmword ptr [r14 + 64]
    vfmadd231ps xmm10, xmm9, xmm4
    vmovups xmmword ptr [r14 + 64], xmm10
    add r13, 192
    add r10, 192
    add r14, 192
    add r11, 4
    dec rbx
    jnz 15b
    vmovups xmmword ptr [rdi + r12*1 + 48], xmm7
    inc rax
    cmp rax, r8
    jb 13b
16:
    pop r14
    pop r13
    pop r12
    pop rbx
    vzeroupper
    ret
# c0, aligned to 4:
#   00 00 00 3f
