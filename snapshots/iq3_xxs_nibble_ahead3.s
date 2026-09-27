# iq3_xxs_nibble_ahead3(gpr, gpr, gpr, gpr) -> vec
# frame: 64 bytes aligned to 64, saves []
# row=rdi sbs=rsi quants=rdx scales=rcx levels=ymm0 nibble=ymm1
# one16=ymm2 acc=ymm3 cur=rax nxt=r8 p0=ymm4 p1=ymm5
# p2=ymm6 p3=ymm8 p4=ymm9 p5=ymm5 p6=ymm10 p7=ymm8
# d=ymm4,ymm7 steps=ymm5,ymm4 scale=ymm5,ymm6 packed=ymm4,ymm5,ymm6,ymm8,ymm9,ymm7 low_codes=ymm5,ymm6,ymm7,ymm9,ymm10,ymm8 unsigned_levels=ymm5,ymm6,ymm7,ymm9,ymm10,ymm4,ymm0
# x=ymm6,ymm7,ymm8,ymm10,ymm11,ymm9,ymm1 lanes=ymm4,ymm5,ymm6,ymm7,ymm0 group_scale=ymm9,ymm10,ymm8,ymm7,ymm5,ymm6,ymm1 zero=ymm0 high=ymm0 sum=ymm0
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
    vmovd xmm5, dword ptr [rdi + 2]
    vpshufb xmm5, xmm5, xmmword ptr [rip + {c3}]
    vpmovzxbd ymm5, xmm5
    vpsrlvd ymm5, ymm5, ymmword ptr [rip + {c4}]
    vpand ymm5, ymm5, ymmword ptr [rip + {c5}]
    vpaddd ymm5, ymm5, ymm5
    vpor ymm5, ymm5, ymmword ptr [rip + {c6}]
    vcvtdq2ps ymm5, ymm5
    vmulps ymm5, ymm4, ymm5
    vmulps ymm5, ymm5, ymmword ptr [rip + {c7}]
    vmulps ymm5, ymm5, ymmword ptr [rcx]
    vmovaps ymmword ptr [rax], ymm5
    # products of group 0 of superblock +0
    vmovdqu xmm4, xmmword ptr [rdi + 6]
    vpand xmm5, xmm4, xmm1
    vpsrlw xmm4, xmm4, 4
    vpand xmm4, xmm4, xmm1
    vinserti128 ymm4, ymm5, xmm4, 1
    vpshufb ymm4, ymm0, ymm4
    vpabsb ymm5, ymm4
    vmovdqa ymm6, ymmword ptr [rdx]
    vpsignb ymm6, ymm6, ymm4
    vpmaddubsw ymm4, ymm5, ymm6
    # products of group 1 of superblock +0
    vmovdqu xmm5, xmmword ptr [rdi + 22]
    vpand xmm6, xmm5, xmm1
    vpsrlw xmm5, xmm5, 4
    vpand xmm5, xmm5, xmm1
    vinserti128 ymm5, ymm6, xmm5, 1
    vpshufb ymm5, ymm0, ymm5
    vpabsb ymm6, ymm5
    vmovdqa ymm7, ymmword ptr [rdx + 32]
    vpsignb ymm7, ymm7, ymm5
    vpmaddubsw ymm5, ymm6, ymm7
    # products of group 2 of superblock +0
    vmovdqu xmm6, xmmword ptr [rdi + 38]
    vpand xmm7, xmm6, xmm1
    vpsrlw xmm6, xmm6, 4
    vpand xmm6, xmm6, xmm1
    vinserti128 ymm6, ymm7, xmm6, 1
    vpshufb ymm6, ymm0, ymm6
    vpabsb ymm7, ymm6
    vmovdqa ymm8, ymmword ptr [rdx + 64]
    vpsignb ymm8, ymm8, ymm6
    vpmaddubsw ymm6, ymm7, ymm8
    dec rsi
    jz 3f
2:
    # scales of superblock +1
    vpbroadcastw xmm7, word ptr [rdi + 134]
    vcvtph2ps ymm7, xmm7
    # products of group 3 of superblock +0
    vmovdqu xmm8, xmmword ptr [rdi + 54]
    vpand xmm9, xmm8, xmm1
    vpsrlw xmm8, xmm8, 4
    vpand xmm8, xmm8, xmm1
    vinserti128 ymm8, ymm9, xmm8, 1
    vpshufb ymm8, ymm0, ymm8
    vpabsb ymm9, ymm8
    vmovdqa ymm10, ymmword ptr [rdx + 96]
    vpsignb ymm10, ymm10, ymm8
    vpmaddubsw ymm8, ymm9, ymm10
    # sum of group 0
    vpmaddwd ymm4, ymm4, ymm2
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm9, dword ptr [rax]
    vfmadd231ps ymm3, ymm4, ymm9
    vmovd xmm4, dword ptr [rdi + 136]
    vpshufb xmm4, xmm4, xmmword ptr [rip + {c3}]
    # products of group 4 of superblock +0
    vmovdqu xmm9, xmmword ptr [rdi + 70]
    vpand xmm10, xmm9, xmm1
    vpsrlw xmm9, xmm9, 4
    vpand xmm9, xmm9, xmm1
    vinserti128 ymm9, ymm10, xmm9, 1
    vpshufb ymm9, ymm0, ymm9
    vpabsb ymm10, ymm9
    vmovdqa ymm11, ymmword ptr [rdx + 128]
    vpsignb ymm11, ymm11, ymm9
    vpmaddubsw ymm9, ymm10, ymm11
    # sum of group 1
    vpmaddwd ymm5, ymm5, ymm2
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm10, dword ptr [rax + 4]
    vfmadd231ps ymm3, ymm5, ymm10
    vpmovzxbd ymm4, xmm4
    vpsrlvd ymm4, ymm4, ymmword ptr [rip + {c4}]
    # products of group 5 of superblock +0
    vmovdqu xmm5, xmmword ptr [rdi + 86]
    vpand xmm10, xmm5, xmm1
    vpsrlw xmm5, xmm5, 4
    vpand xmm5, xmm5, xmm1
    vinserti128 ymm5, ymm10, xmm5, 1
    vpshufb ymm5, ymm0, ymm5
    vpabsb ymm10, ymm5
    vmovdqa ymm11, ymmword ptr [rdx + 160]
    vpsignb ymm11, ymm11, ymm5
    vpmaddubsw ymm5, ymm10, ymm11
    # sum of group 2
    vpmaddwd ymm6, ymm6, ymm2
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm10, dword ptr [rax + 8]
    vfmadd231ps ymm3, ymm6, ymm10
    vpand ymm4, ymm4, ymmword ptr [rip + {c5}]
    vpaddd ymm6, ymm4, ymm4
    # products of group 6 of superblock +0
    vmovdqu xmm4, xmmword ptr [rdi + 102]
    vpand xmm10, xmm4, xmm1
    vpsrlw xmm4, xmm4, 4
    vpand xmm4, xmm4, xmm1
    vinserti128 ymm10, ymm10, xmm4, 1
    vpshufb ymm10, ymm0, ymm10
    vpabsb ymm4, ymm10
    vmovdqa ymm11, ymmword ptr [rdx + 192]
    vpsignb ymm11, ymm11, ymm10
    vpmaddubsw ymm10, ymm4, ymm11
    # sum of group 3
    vpmaddwd ymm4, ymm8, ymm2
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm8, dword ptr [rax + 12]
    vfmadd231ps ymm3, ymm4, ymm8
    vpor ymm6, ymm6, ymmword ptr [rip + {c6}]
    vcvtdq2ps ymm6, ymm6
    # products of group 7 of superblock +0
    vmovdqu xmm4, xmmword ptr [rdi + 118]
    vpand xmm8, xmm4, xmm1
    vpsrlw xmm4, xmm4, 4
    vpand xmm4, xmm4, xmm1
    vinserti128 ymm8, ymm8, xmm4, 1
    vpshufb ymm8, ymm0, ymm8
    vpabsb ymm4, ymm8
    vmovdqa ymm11, ymmword ptr [rdx + 224]
    vpsignb ymm11, ymm11, ymm8
    vpmaddubsw ymm8, ymm4, ymm11
    # sum of group 4
    vpmaddwd ymm4, ymm9, ymm2
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm9, dword ptr [rax + 16]
    vfmadd231ps ymm3, ymm4, ymm9
    vmulps ymm6, ymm7, ymm6
    vmulps ymm6, ymm6, ymmword ptr [rip + {c7}]
    # products of group 0 of superblock +1
    vmovdqu xmm4, xmmword ptr [rdi + 140]
    vpand xmm7, xmm4, xmm1
    vpsrlw xmm4, xmm4, 4
    vpand xmm4, xmm4, xmm1
    vinserti128 ymm4, ymm7, xmm4, 1
    vpshufb ymm4, ymm0, ymm4
    vpabsb ymm7, ymm4
    vmovdqa ymm9, ymmword ptr [rdx + 256]
    vpsignb ymm9, ymm9, ymm4
    vpmaddubsw ymm4, ymm7, ymm9
    # sum of group 5
    vpmaddwd ymm5, ymm5, ymm2
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm7, dword ptr [rax + 20]
    vfmadd231ps ymm3, ymm5, ymm7
    vmulps ymm6, ymm6, ymmword ptr [rcx + 32]
    vmovaps ymmword ptr [r8], ymm6
    # products of group 1 of superblock +1
    vmovdqu xmm5, xmmword ptr [rdi + 156]
    vpand xmm6, xmm5, xmm1
    vpsrlw xmm5, xmm5, 4
    vpand xmm5, xmm5, xmm1
    vinserti128 ymm5, ymm6, xmm5, 1
    vpshufb ymm5, ymm0, ymm5
    vpabsb ymm6, ymm5
    vmovdqa ymm7, ymmword ptr [rdx + 288]
    vpsignb ymm7, ymm7, ymm5
    vpmaddubsw ymm5, ymm6, ymm7
    # sum of group 6
    vpmaddwd ymm6, ymm10, ymm2
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 24]
    vfmadd231ps ymm3, ymm6, ymm7
    # products of group 2 of superblock +1
    vmovdqu xmm6, xmmword ptr [rdi + 172]
    vpand xmm7, xmm6, xmm1
    vpsrlw xmm6, xmm6, 4
    vpand xmm6, xmm6, xmm1
    vinserti128 ymm6, ymm7, xmm6, 1
    vpshufb ymm6, ymm0, ymm6
    vpabsb ymm7, ymm6
    vmovdqa ymm9, ymmword ptr [rdx + 320]
    vpsignb ymm9, ymm9, ymm6
    vpmaddubsw ymm6, ymm7, ymm9
    # sum of group 7
    vpmaddwd ymm7, ymm8, ymm2
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rax + 28]
    vfmadd231ps ymm3, ymm7, ymm8
    add rdi, 134
    add rdx, 256
    add rcx, 32
    xor rax, 32
    xor r8, 32
    dec rsi
    jnz 2b
3:
    # products of group 3 of superblock +0
    vmovdqu xmm7, xmmword ptr [rdi + 54]
    vpand xmm8, xmm7, xmm1
    vpsrlw xmm7, xmm7, 4
    vpand xmm7, xmm7, xmm1
    vinserti128 ymm8, ymm8, xmm7, 1
    vpshufb ymm8, ymm0, ymm8
    vpabsb ymm7, ymm8
    vmovdqa ymm9, ymmword ptr [rdx + 96]
    vpsignb ymm9, ymm9, ymm8
    vpmaddubsw ymm8, ymm7, ymm9
    # sum of group 0
    vpmaddwd ymm4, ymm4, ymm2
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm7, dword ptr [rax]
    vfmadd231ps ymm3, ymm4, ymm7
    # products of group 4 of superblock +0
    vmovdqu xmm4, xmmword ptr [rdi + 70]
    vpand xmm7, xmm4, xmm1
    vpsrlw xmm4, xmm4, 4
    vpand xmm4, xmm4, xmm1
    vinserti128 ymm9, ymm7, xmm4, 1
    vpshufb ymm9, ymm0, ymm9
    vpabsb ymm4, ymm9
    vmovdqa ymm7, ymmword ptr [rdx + 128]
    vpsignb ymm7, ymm7, ymm9
    vpmaddubsw ymm9, ymm4, ymm7
    # sum of group 1
    vpmaddwd ymm4, ymm5, ymm2
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, dword ptr [rax + 4]
    vfmadd231ps ymm3, ymm4, ymm5
    # products of group 5 of superblock +0
    vmovdqu xmm4, xmmword ptr [rdi + 86]
    vpand xmm5, xmm4, xmm1
    vpsrlw xmm4, xmm4, 4
    vpand xmm4, xmm4, xmm1
    vinserti128 ymm5, ymm5, xmm4, 1
    vpshufb ymm5, ymm0, ymm5
    vpabsb ymm4, ymm5
    vmovdqa ymm7, ymmword ptr [rdx + 160]
    vpsignb ymm7, ymm7, ymm5
    vpmaddubsw ymm5, ymm4, ymm7
    # sum of group 2
    vpmaddwd ymm4, ymm6, ymm2
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm6, dword ptr [rax + 8]
    vfmadd231ps ymm3, ymm4, ymm6
    # products of group 6 of superblock +0
    vmovdqu xmm4, xmmword ptr [rdi + 102]
    vpand xmm6, xmm4, xmm1
    vpsrlw xmm4, xmm4, 4
    vpand xmm4, xmm4, xmm1
    vinserti128 ymm10, ymm6, xmm4, 1
    vpshufb ymm10, ymm0, ymm10
    vpabsb ymm4, ymm10
    vmovdqa ymm6, ymmword ptr [rdx + 192]
    vpsignb ymm6, ymm6, ymm10
    vpmaddubsw ymm10, ymm4, ymm6
    # sum of group 3
    vpmaddwd ymm4, ymm8, ymm2
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm6, dword ptr [rax + 12]
    vfmadd231ps ymm3, ymm4, ymm6
    # products of group 7 of superblock +0
    vmovdqu xmm4, xmmword ptr [rdi + 118]
    vpand xmm6, xmm4, xmm1
    vpsrlw xmm4, xmm4, 4
    vpand xmm4, xmm4, xmm1
    vinserti128 ymm8, ymm6, xmm4, 1
    vpshufb ymm8, ymm0, ymm8
    vpabsb ymm0, ymm8
    vmovdqa ymm1, ymmword ptr [rdx + 224]
    vpsignb ymm1, ymm1, ymm8
    vpmaddubsw ymm8, ymm0, ymm1
    # sum of group 4
    vpmaddwd ymm0, ymm9, ymm2
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm1, dword ptr [rax + 16]
    vfmadd231ps ymm3, ymm0, ymm1
    # sum of group 5
    vpmaddwd ymm0, ymm5, ymm2
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm1, dword ptr [rax + 20]
    vfmadd231ps ymm3, ymm0, ymm1
    # sum of group 6
    vpmaddwd ymm0, ymm10, ymm2
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm1, dword ptr [rax + 24]
    vfmadd231ps ymm3, ymm0, ymm1
    # sum of group 7
    vpmaddwd ymm0, ymm8, ymm2
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
#   04 0c 14 1c 24 2c 34 3e fc f4 ec e4 dc d4 cc c2
#   04 0c 14 1c 24 2c 34 3e fc f4 ec e4 dc d4 cc c2
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
#   01 00 00 00 01 00 00 00 01 00 00 00 01 00 00 00
#   01 00 00 00 01 00 00 00 01 00 00 00 01 00 00 00
# c7, aligned to 32:
#   00 00 80 3e 00 00 80 3e 00 00 80 3e 00 00 80 3e
#   00 00 80 3e 00 00 80 3e 00 00 80 3e 00 00 80 3e
