# iq4_xs_ahead1(gpr, gpr, gpr, gpr) -> vec
# frame: 64 bytes aligned to 64, saves []
# row=rdi sbs=rsi quants=rdx scales=rcx levels=ymm0 nibble=ymm1
# one16=ymm2 acc=ymm3 cur=rax nxt=r8 p0=ymm4 p1=ymm7
# p2=ymm4 p3=ymm8 p4=ymm4 p5=ymm5 p6=ymm4 p7=ymm5
# d=ymm4,ymm5 steps=ymm5,ymm6 scale=ymm6,ymm7 packed=ymm4,ymm7,ymm8,ymm5 low_codes=ymm5,ymm8,ymm9,ymm6 unsigned_levels=ymm5,ymm8,ymm9,ymm6,ymm0
# x=ymm6,ymm9,ymm10,ymm8,ymm7,ymm1 lanes=ymm4,ymm7,ymm6,ymm5,ymm0 group_scale=ymm8,ymm9,ymm6,ymm5,ymm1 zero=ymm0 high=ymm0 sum=ymm0
# pair=ymm1 result=ymm0 nothing=ymm0
    push rbp
    mov rbp, rsp
    and rsp, -64
    sub rsp, 64
    test rsi, rsi
    jz 4f
    # constants
    vmovdqa ymm0, ymmword ptr [rip + {c0}]
    vpbroadcastd ymm1, dword ptr [rip + {c1}]
    vpbroadcastd ymm2, dword ptr [rip + {c2}]
    vxorps xmm3, xmm3, xmm3
    lea rax, [rsp]
    lea r8, [rsp + 32]
    # scales of superblock +0
    vpbroadcastw xmm4, word ptr [rdi]
    vcvtph2ps ymm4, xmm4
    vmovd xmm5, dword ptr [rdi + 4]
    vpshufb xmm5, xmm5, xmmword ptr [rip + {c3}]
    vpmovzxbd ymm5, xmm5
    vpsrlvd ymm5, ymm5, ymmword ptr [rip + {c4}]
    vpand ymm5, ymm5, ymmword ptr [rip + {c5}]
    vpbroadcastw ymm6, word ptr [rdi + 2]
    vpsrlvd ymm6, ymm6, ymmword ptr [rip + {c6}]
    vpand ymm6, ymm6, ymmword ptr [rip + {c7}]
    vpslld ymm6, ymm6, 4
    vpor ymm6, ymm6, ymm5
    vpaddd ymm6, ymm6, ymmword ptr [rip + {c8}]
    vcvtdq2ps ymm6, ymm6
    vmulps ymm6, ymm4, ymm6
    vmulps ymm6, ymm6, ymmword ptr [rcx]
    vmovaps ymmword ptr [rax], ymm6
    # products of group 0 of superblock +0
    vmovdqu xmm4, xmmword ptr [rdi + 8]
    vpand xmm5, xmm4, xmm1
    vpsrlw xmm4, xmm4, 4
    vpand xmm4, xmm4, xmm1
    vinserti128 ymm4, ymm5, xmm4, 1
    vpshufb ymm4, ymm0, ymm4
    vpabsb ymm5, ymm4
    vmovdqa ymm6, ymmword ptr [rdx]
    vpsignb ymm6, ymm6, ymm4
    vpmaddubsw ymm4, ymm5, ymm6
    dec rsi
    jz 3f
2:
    # scales of superblock +1
    vpbroadcastw xmm5, word ptr [rdi + 136]
    vcvtph2ps ymm5, xmm5
    vmovd xmm6, dword ptr [rdi + 140]
    # products of group 1 of superblock +0
    vmovdqu xmm7, xmmword ptr [rdi + 24]
    vpand xmm8, xmm7, xmm1
    vpsrlw xmm7, xmm7, 4
    vpand xmm7, xmm7, xmm1
    vinserti128 ymm7, ymm8, xmm7, 1
    vpshufb ymm7, ymm0, ymm7
    vpabsb ymm8, ymm7
    vmovdqa ymm9, ymmword ptr [rdx + 32]
    vpsignb ymm9, ymm9, ymm7
    vpmaddubsw ymm7, ymm8, ymm9
    # sum of group 0
    vpmaddwd ymm4, ymm4, ymm2
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm8, dword ptr [rax]
    vfmadd231ps ymm3, ymm4, ymm8
    vpshufb xmm6, xmm6, xmmword ptr [rip + {c3}]
    vpmovzxbd ymm6, xmm6
    vpsrlvd ymm6, ymm6, ymmword ptr [rip + {c4}]
    # products of group 2 of superblock +0
    vmovdqu xmm4, xmmword ptr [rdi + 40]
    vpand xmm8, xmm4, xmm1
    vpsrlw xmm4, xmm4, 4
    vpand xmm4, xmm4, xmm1
    vinserti128 ymm4, ymm8, xmm4, 1
    vpshufb ymm4, ymm0, ymm4
    vpabsb ymm8, ymm4
    vmovdqa ymm9, ymmword ptr [rdx + 64]
    vpsignb ymm9, ymm9, ymm4
    vpmaddubsw ymm4, ymm8, ymm9
    # sum of group 1
    vpmaddwd ymm7, ymm7, ymm2
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rax + 4]
    vfmadd231ps ymm3, ymm7, ymm8
    vpand ymm6, ymm6, ymmword ptr [rip + {c5}]
    vpbroadcastw ymm7, word ptr [rdi + 138]
    vpsrlvd ymm7, ymm7, ymmword ptr [rip + {c6}]
    # products of group 3 of superblock +0
    vmovdqu xmm8, xmmword ptr [rdi + 56]
    vpand xmm9, xmm8, xmm1
    vpsrlw xmm8, xmm8, 4
    vpand xmm8, xmm8, xmm1
    vinserti128 ymm8, ymm9, xmm8, 1
    vpshufb ymm8, ymm0, ymm8
    vpabsb ymm9, ymm8
    vmovdqa ymm10, ymmword ptr [rdx + 96]
    vpsignb ymm10, ymm10, ymm8
    vpmaddubsw ymm8, ymm9, ymm10
    # sum of group 2
    vpmaddwd ymm4, ymm4, ymm2
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm9, dword ptr [rax + 8]
    vfmadd231ps ymm3, ymm4, ymm9
    vpand ymm7, ymm7, ymmword ptr [rip + {c7}]
    vpslld ymm7, ymm7, 4
    vpor ymm7, ymm7, ymm6
    # products of group 4 of superblock +0
    vmovdqu xmm4, xmmword ptr [rdi + 72]
    vpand xmm6, xmm4, xmm1
    vpsrlw xmm4, xmm4, 4
    vpand xmm4, xmm4, xmm1
    vinserti128 ymm4, ymm6, xmm4, 1
    vpshufb ymm4, ymm0, ymm4
    vpabsb ymm6, ymm4
    vmovdqa ymm9, ymmword ptr [rdx + 128]
    vpsignb ymm9, ymm9, ymm4
    vpmaddubsw ymm4, ymm6, ymm9
    # sum of group 3
    vpmaddwd ymm6, ymm8, ymm2
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm8, dword ptr [rax + 12]
    vfmadd231ps ymm3, ymm6, ymm8
    vpaddd ymm7, ymm7, ymmword ptr [rip + {c8}]
    vcvtdq2ps ymm7, ymm7
    vmulps ymm7, ymm5, ymm7
    # products of group 5 of superblock +0
    vmovdqu xmm5, xmmword ptr [rdi + 88]
    vpand xmm6, xmm5, xmm1
    vpsrlw xmm5, xmm5, 4
    vpand xmm5, xmm5, xmm1
    vinserti128 ymm5, ymm6, xmm5, 1
    vpshufb ymm5, ymm0, ymm5
    vpabsb ymm6, ymm5
    vmovdqa ymm8, ymmword ptr [rdx + 160]
    vpsignb ymm8, ymm8, ymm5
    vpmaddubsw ymm5, ymm6, ymm8
    # sum of group 4
    vpmaddwd ymm4, ymm4, ymm2
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm6, dword ptr [rax + 16]
    vfmadd231ps ymm3, ymm4, ymm6
    vmulps ymm7, ymm7, ymmword ptr [rcx + 32]
    vmovaps ymmword ptr [r8], ymm7
    # products of group 6 of superblock +0
    vmovdqu xmm4, xmmword ptr [rdi + 104]
    vpand xmm6, xmm4, xmm1
    vpsrlw xmm4, xmm4, 4
    vpand xmm4, xmm4, xmm1
    vinserti128 ymm4, ymm6, xmm4, 1
    vpshufb ymm4, ymm0, ymm4
    vpabsb ymm6, ymm4
    vmovdqa ymm7, ymmword ptr [rdx + 192]
    vpsignb ymm7, ymm7, ymm4
    vpmaddubsw ymm4, ymm6, ymm7
    # sum of group 5
    vpmaddwd ymm5, ymm5, ymm2
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 20]
    vfmadd231ps ymm3, ymm5, ymm6
    # products of group 7 of superblock +0
    vmovdqu xmm5, xmmword ptr [rdi + 120]
    vpand xmm6, xmm5, xmm1
    vpsrlw xmm5, xmm5, 4
    vpand xmm5, xmm5, xmm1
    vinserti128 ymm5, ymm6, xmm5, 1
    vpshufb ymm5, ymm0, ymm5
    vpabsb ymm6, ymm5
    vmovdqa ymm7, ymmword ptr [rdx + 224]
    vpsignb ymm7, ymm7, ymm5
    vpmaddubsw ymm5, ymm6, ymm7
    # sum of group 6
    vpmaddwd ymm4, ymm4, ymm2
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm6, dword ptr [rax + 24]
    vfmadd231ps ymm3, ymm4, ymm6
    # products of group 0 of superblock +1
    vmovdqu xmm4, xmmword ptr [rdi + 144]
    vpand xmm6, xmm4, xmm1
    vpsrlw xmm4, xmm4, 4
    vpand xmm4, xmm4, xmm1
    vinserti128 ymm4, ymm6, xmm4, 1
    vpshufb ymm4, ymm0, ymm4
    vpabsb ymm6, ymm4
    vmovdqa ymm7, ymmword ptr [rdx + 256]
    vpsignb ymm7, ymm7, ymm4
    vpmaddubsw ymm4, ymm6, ymm7
    # sum of group 7
    vpmaddwd ymm5, ymm5, ymm2
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 28]
    vfmadd231ps ymm3, ymm5, ymm6
    add rdi, 136
    add rdx, 256
    add rcx, 32
    xor rax, 32
    xor r8, 32
    dec rsi
    jnz 2b
3:
    # products of group 1 of superblock +0
    vmovdqu xmm5, xmmword ptr [rdi + 24]
    vpand xmm6, xmm5, xmm1
    vpsrlw xmm5, xmm5, 4
    vpand xmm5, xmm5, xmm1
    vinserti128 ymm7, ymm6, xmm5, 1
    vpshufb ymm7, ymm0, ymm7
    vpabsb ymm5, ymm7
    vmovdqa ymm6, ymmword ptr [rdx + 32]
    vpsignb ymm6, ymm6, ymm7
    vpmaddubsw ymm7, ymm5, ymm6
    # sum of group 0
    vpmaddwd ymm4, ymm4, ymm2
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, dword ptr [rax]
    vfmadd231ps ymm3, ymm4, ymm5
    # products of group 2 of superblock +0
    vmovdqu xmm4, xmmword ptr [rdi + 40]
    vpand xmm5, xmm4, xmm1
    vpsrlw xmm4, xmm4, 4
    vpand xmm4, xmm4, xmm1
    vinserti128 ymm4, ymm5, xmm4, 1
    vpshufb ymm4, ymm0, ymm4
    vpabsb ymm5, ymm4
    vmovdqa ymm6, ymmword ptr [rdx + 64]
    vpsignb ymm6, ymm6, ymm4
    vpmaddubsw ymm4, ymm5, ymm6
    # sum of group 1
    vpmaddwd ymm5, ymm7, ymm2
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 4]
    vfmadd231ps ymm3, ymm5, ymm6
    # products of group 3 of superblock +0
    vmovdqu xmm5, xmmword ptr [rdi + 56]
    vpand xmm6, xmm5, xmm1
    vpsrlw xmm5, xmm5, 4
    vpand xmm5, xmm5, xmm1
    vinserti128 ymm8, ymm6, xmm5, 1
    vpshufb ymm8, ymm0, ymm8
    vpabsb ymm5, ymm8
    vmovdqa ymm6, ymmword ptr [rdx + 96]
    vpsignb ymm6, ymm6, ymm8
    vpmaddubsw ymm8, ymm5, ymm6
    # sum of group 2
    vpmaddwd ymm4, ymm4, ymm2
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, dword ptr [rax + 8]
    vfmadd231ps ymm3, ymm4, ymm5
    # products of group 4 of superblock +0
    vmovdqu xmm4, xmmword ptr [rdi + 72]
    vpand xmm5, xmm4, xmm1
    vpsrlw xmm4, xmm4, 4
    vpand xmm4, xmm4, xmm1
    vinserti128 ymm4, ymm5, xmm4, 1
    vpshufb ymm4, ymm0, ymm4
    vpabsb ymm5, ymm4
    vmovdqa ymm6, ymmword ptr [rdx + 128]
    vpsignb ymm6, ymm6, ymm4
    vpmaddubsw ymm4, ymm5, ymm6
    # sum of group 3
    vpmaddwd ymm5, ymm8, ymm2
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 12]
    vfmadd231ps ymm3, ymm5, ymm6
    # products of group 5 of superblock +0
    vmovdqu xmm5, xmmword ptr [rdi + 88]
    vpand xmm6, xmm5, xmm1
    vpsrlw xmm5, xmm5, 4
    vpand xmm5, xmm5, xmm1
    vinserti128 ymm5, ymm6, xmm5, 1
    vpshufb ymm5, ymm0, ymm5
    vpabsb ymm6, ymm5
    vmovdqa ymm7, ymmword ptr [rdx + 160]
    vpsignb ymm7, ymm7, ymm5
    vpmaddubsw ymm5, ymm6, ymm7
    # sum of group 4
    vpmaddwd ymm4, ymm4, ymm2
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm6, dword ptr [rax + 16]
    vfmadd231ps ymm3, ymm4, ymm6
    # products of group 6 of superblock +0
    vmovdqu xmm4, xmmword ptr [rdi + 104]
    vpand xmm6, xmm4, xmm1
    vpsrlw xmm4, xmm4, 4
    vpand xmm4, xmm4, xmm1
    vinserti128 ymm4, ymm6, xmm4, 1
    vpshufb ymm4, ymm0, ymm4
    vpabsb ymm6, ymm4
    vmovdqa ymm7, ymmword ptr [rdx + 192]
    vpsignb ymm7, ymm7, ymm4
    vpmaddubsw ymm4, ymm6, ymm7
    # sum of group 5
    vpmaddwd ymm5, ymm5, ymm2
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 20]
    vfmadd231ps ymm3, ymm5, ymm6
    # products of group 7 of superblock +0
    vmovdqu xmm5, xmmword ptr [rdi + 120]
    vpand xmm6, xmm5, xmm1
    vpsrlw xmm5, xmm5, 4
    vpand xmm5, xmm5, xmm1
    vinserti128 ymm5, ymm6, xmm5, 1
    vpshufb ymm5, ymm0, ymm5
    vpabsb ymm0, ymm5
    vmovdqa ymm1, ymmword ptr [rdx + 224]
    vpsignb ymm1, ymm1, ymm5
    vpmaddubsw ymm5, ymm0, ymm1
    # sum of group 6
    vpmaddwd ymm0, ymm4, ymm2
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm1, dword ptr [rax + 24]
    vfmadd231ps ymm3, ymm0, ymm1
    # sum of group 7
    vpmaddwd ymm0, ymm5, ymm2
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm1, dword ptr [rax + 28]
    vfmadd231ps ymm3, ymm0, ymm1
    # the lanes summed
    vxorps xmm0, xmm0, xmm0
    vaddps ymm3, ymm3, ymm0
    vextractf128 xmm0, ymm3, 1
    vaddps xmm0, xmm3, xmm0
    vmovhlps xmm1, xmm0, xmm0
    vaddps xmm0, xmm0, xmm1
    vmovshdup xmm1, xmm0
    vaddss xmm0, xmm0, xmm1
    mov rsp, rbp
    pop rbp
    vzeroupper
    ret
4:
    vxorps xmm0, xmm0, xmm0
    mov rsp, rbp
    pop rbp
    vzeroupper
    ret
# c0, aligned to 32:
#   81 98 ad bf cf dd ea f6 01 0d 19 26 35 45 59 71
#   81 98 ad bf cf dd ea f6 01 0d 19 26 35 45 59 71
# c1, aligned to 4:
#   0f 0f 0f 0f
# c2, aligned to 4:
#   01 00 01 00
# c3, aligned to 16:
#   00 00 01 01 02 02 03 03 80 80 80 80 80 80 80 80
# c4, aligned to 32:
#   00 00 00 00 04 00 00 00 00 00 00 00 04 00 00 00
#   00 00 00 00 04 00 00 00 00 00 00 00 04 00 00 00
# c5, aligned to 32:
#   0f 00 00 00 0f 00 00 00 0f 00 00 00 0f 00 00 00
#   0f 00 00 00 0f 00 00 00 0f 00 00 00 0f 00 00 00
# c6, aligned to 32:
#   00 00 00 00 02 00 00 00 04 00 00 00 06 00 00 00
#   08 00 00 00 0a 00 00 00 0c 00 00 00 0e 00 00 00
# c7, aligned to 32:
#   03 00 00 00 03 00 00 00 03 00 00 00 03 00 00 00
#   03 00 00 00 03 00 00 00 03 00 00 00 03 00 00 00
# c8, aligned to 32:
#   e0 ff ff ff e0 ff ff ff e0 ff ff ff e0 ff ff ff
#   e0 ff ff ff e0 ff ff ff e0 ff ff ff e0 ff ff ff
