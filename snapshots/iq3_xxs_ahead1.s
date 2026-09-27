# iq3_xxs_ahead1(gpr, gpr, gpr, gpr, gpr) -> vec
# frame: 128 bytes aligned to 128, saves []
# row=rdi sbs=rsi quants=rdx scales=rcx grid=r8 spread=ymm0
# bit=ymm1 max8=ymm2 one16=ymm3 acc=ymm4 cur=rax nxt=r9
# p0=ymm7 p1=ymm11 p2=ymm10 p3=ymm11 p4=ymm9 p5=ymm12
# p6=ymm10 p7=ymm8 steps=ymm5 words=ymm5 fields=ymm6 field=ymm7,ymm8
# parity=ymm8,ymm7 d=ymm6 scale=ymm5 indices=ymm5,ymm9,ymm7,ymm10,ymm8,ymm6 gather_mask=ymm6,ymm10,ymm9,ymm8,ymm11,ymm7 signs=ymm5,ymm9,ymm7,ymm10,ymm8,ymm6
# x=ymm6,ymm10,ymm9,ymm8,ymm11,ymm7,ymm0 lanes=ymm7,ymm9,ymm8,ymm6,ymm5,ymm0 group_scale=ymm9,ymm8,ymm10,ymm7,ymm6,ymm1 zero=ymm0 high=ymm0 sum=ymm0
# pair=ymm1 result=ymm0 nothing=ymm0
    push rbp
    mov rbp, rsp
    and rsp, -128
    sub rsp, 128
    test rsi, rsi
    jz 4f
    # constants
    vmovdqa ymm0, ymmword ptr [rip + {c0}]
    vpbroadcastq ymm1, qword ptr [rip + {c1}]
    vpbroadcastd ymm2, dword ptr [rip + {c2}]
    vpbroadcastd ymm3, dword ptr [rip + {c3}]
    vxorps xmm4, xmm4, xmm4
    lea rax, [rsp]
    lea r9, [rsp + 64]
    # signs of superblock +0
    vmovdqu ymm5, ymmword ptr [rdi + 66]
    vpand ymm6, ymm5, ymmword ptr [rip + {c4}]
    vpaddd ymm7, ymm5, ymm5
    vpand ymm7, ymm7, ymmword ptr [rip + {c5}]
    vpor ymm6, ymm6, ymm7
    vpslld ymm7, ymm5, 2
    vpand ymm7, ymm7, ymmword ptr [rip + {c6}]
    vpor ymm6, ymm6, ymm7
    vpslld ymm7, ymm5, 3
    vpand ymm7, ymm7, ymmword ptr [rip + {c7}]
    vpor ymm6, ymm6, ymm7
    vpsrld ymm5, ymm5, 27
    vpaddw ymm8, ymm6, ymm6
    vpxor ymm8, ymm8, ymm6
    vpsllw ymm7, ymm8, 2
    vpxor ymm8, ymm8, ymm7
    vpsllw ymm7, ymm8, 4
    vpxor ymm8, ymm8, ymm7
    vpand ymm8, ymm8, ymmword ptr [rip + {c8}]
    vpor ymm6, ymm6, ymm8
    vmovdqa ymmword ptr [rax + 32], ymm6
    # scales of superblock +0
    vpbroadcastw xmm6, word ptr [rdi]
    vcvtph2ps ymm6, xmm6
    vpor ymm5, ymm5, ymmword ptr [rip + {c9}]
    vcvtdq2ps ymm5, ymm5
    vmulps ymm5, ymm6, ymm5
    vmulps ymm5, ymm5, ymmword ptr [rip + {c10}]
    vmulps ymm5, ymm5, ymmword ptr [rcx]
    vmovaps ymmword ptr [rax], ymm5
    # products of group 0 of superblock +0
    vpmovzxbd ymm5, qword ptr [rdi + 2]
    vpcmpeqd ymm6, ymm6, ymm6
    vxorps xmm7, xmm7, xmm7
    vpgatherdd ymm7, dword ptr [r8 + ymm5*4], ymm6
    vpbroadcastd ymm5, dword ptr [rax + 32]
    vpshufb ymm5, ymm5, ymm0
    vpand ymm5, ymm5, ymm1
    vpaddb ymm5, ymm5, ymm2
    vmovdqa ymm6, ymmword ptr [rdx]
    vpsignb ymm6, ymm6, ymm5
    vpmaddubsw ymm7, ymm7, ymm6
    dec rsi
    jz 3f
2:
    # signs of superblock +1
    vmovdqu ymm5, ymmword ptr [rdi + 164]
    vpand ymm6, ymm5, ymmword ptr [rip + {c4}]
    vpaddd ymm8, ymm5, ymm5
    # products of group 1 of superblock +0
    vpmovzxbd ymm9, qword ptr [rdi + 10]
    vpcmpeqd ymm10, ymm10, ymm10
    vxorps xmm11, xmm11, xmm11
    vpgatherdd ymm11, dword ptr [r8 + ymm9*4], ymm10
    vpbroadcastd ymm9, dword ptr [rax + 36]
    vpshufb ymm9, ymm9, ymm0
    vpand ymm9, ymm9, ymm1
    vpaddb ymm9, ymm9, ymm2
    vmovdqa ymm10, ymmword ptr [rdx + 32]
    vpsignb ymm10, ymm10, ymm9
    vpmaddubsw ymm11, ymm11, ymm10
    # sum of group 0
    vpmaddwd ymm7, ymm7, ymm3
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm9, dword ptr [rax]
    vfmadd231ps ymm4, ymm7, ymm9
    vpand ymm8, ymm8, ymmword ptr [rip + {c5}]
    vpor ymm6, ymm6, ymm8
    vpslld ymm8, ymm5, 2
    # products of group 2 of superblock +0
    vpmovzxbd ymm7, qword ptr [rdi + 18]
    vpcmpeqd ymm9, ymm9, ymm9
    vxorps xmm10, xmm10, xmm10
    vpgatherdd ymm10, dword ptr [r8 + ymm7*4], ymm9
    vpbroadcastd ymm7, dword ptr [rax + 40]
    vpshufb ymm7, ymm7, ymm0
    vpand ymm7, ymm7, ymm1
    vpaddb ymm7, ymm7, ymm2
    vmovdqa ymm9, ymmword ptr [rdx + 64]
    vpsignb ymm9, ymm9, ymm7
    vpmaddubsw ymm10, ymm10, ymm9
    # sum of group 1
    vpmaddwd ymm7, ymm11, ymm3
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm9, dword ptr [rax + 4]
    vfmadd231ps ymm4, ymm7, ymm9
    vpand ymm8, ymm8, ymmword ptr [rip + {c6}]
    vpor ymm6, ymm6, ymm8
    vpslld ymm8, ymm5, 3
    # products of group 3 of superblock +0
    vpmovzxbd ymm7, qword ptr [rdi + 26]
    vpcmpeqd ymm9, ymm9, ymm9
    vxorps xmm11, xmm11, xmm11
    vpgatherdd ymm11, dword ptr [r8 + ymm7*4], ymm9
    vpbroadcastd ymm7, dword ptr [rax + 44]
    vpshufb ymm7, ymm7, ymm0
    vpand ymm7, ymm7, ymm1
    vpaddb ymm7, ymm7, ymm2
    vmovdqa ymm9, ymmword ptr [rdx + 96]
    vpsignb ymm9, ymm9, ymm7
    vpmaddubsw ymm11, ymm11, ymm9
    # sum of group 2
    vpmaddwd ymm7, ymm10, ymm3
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm9, dword ptr [rax + 8]
    vfmadd231ps ymm4, ymm7, ymm9
    vpand ymm8, ymm8, ymmword ptr [rip + {c7}]
    vpor ymm6, ymm6, ymm8
    vpsrld ymm5, ymm5, 27
    # products of group 4 of superblock +0
    vpmovzxbd ymm7, qword ptr [rdi + 34]
    vpcmpeqd ymm8, ymm8, ymm8
    vxorps xmm9, xmm9, xmm9
    vpgatherdd ymm9, dword ptr [r8 + ymm7*4], ymm8
    vpbroadcastd ymm7, dword ptr [rax + 48]
    vpshufb ymm7, ymm7, ymm0
    vpand ymm7, ymm7, ymm1
    vpaddb ymm7, ymm7, ymm2
    vmovdqa ymm8, ymmword ptr [rdx + 128]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm9, ymm9, ymm8
    # sum of group 3
    vpmaddwd ymm7, ymm11, ymm3
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rax + 12]
    vfmadd231ps ymm4, ymm7, ymm8
    vpaddw ymm7, ymm6, ymm6
    vpxor ymm7, ymm7, ymm6
    vpsllw ymm8, ymm7, 2
    # products of group 5 of superblock +0
    vpmovzxbd ymm10, qword ptr [rdi + 42]
    vpcmpeqd ymm11, ymm11, ymm11
    vxorps xmm12, xmm12, xmm12
    vpgatherdd ymm12, dword ptr [r8 + ymm10*4], ymm11
    vpbroadcastd ymm10, dword ptr [rax + 52]
    vpshufb ymm10, ymm10, ymm0
    vpand ymm10, ymm10, ymm1
    vpaddb ymm10, ymm10, ymm2
    vmovdqa ymm11, ymmword ptr [rdx + 160]
    vpsignb ymm11, ymm11, ymm10
    vpmaddubsw ymm12, ymm12, ymm11
    # sum of group 4
    vpmaddwd ymm9, ymm9, ymm3
    vcvtdq2ps ymm9, ymm9
    vbroadcastss ymm10, dword ptr [rax + 16]
    vfmadd231ps ymm4, ymm9, ymm10
    vpxor ymm7, ymm7, ymm8
    vpsllw ymm8, ymm7, 4
    vpxor ymm7, ymm7, ymm8
    # products of group 6 of superblock +0
    vpmovzxbd ymm8, qword ptr [rdi + 50]
    vpcmpeqd ymm9, ymm9, ymm9
    vxorps xmm10, xmm10, xmm10
    vpgatherdd ymm10, dword ptr [r8 + ymm8*4], ymm9
    vpbroadcastd ymm8, dword ptr [rax + 56]
    vpshufb ymm8, ymm8, ymm0
    vpand ymm8, ymm8, ymm1
    vpaddb ymm8, ymm8, ymm2
    vmovdqa ymm9, ymmword ptr [rdx + 192]
    vpsignb ymm9, ymm9, ymm8
    vpmaddubsw ymm10, ymm10, ymm9
    # sum of group 5
    vpmaddwd ymm8, ymm12, ymm3
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm9, dword ptr [rax + 20]
    vfmadd231ps ymm4, ymm8, ymm9
    vpand ymm7, ymm7, ymmword ptr [rip + {c8}]
    vpor ymm6, ymm6, ymm7
    vmovdqa ymmword ptr [r9 + 32], ymm6
    # products of group 7 of superblock +0
    vpmovzxbd ymm6, qword ptr [rdi + 58]
    vpcmpeqd ymm7, ymm7, ymm7
    vxorps xmm8, xmm8, xmm8
    vpgatherdd ymm8, dword ptr [r8 + ymm6*4], ymm7
    vpbroadcastd ymm6, dword ptr [rax + 60]
    vpshufb ymm6, ymm6, ymm0
    vpand ymm6, ymm6, ymm1
    vpaddb ymm6, ymm6, ymm2
    vmovdqa ymm7, ymmword ptr [rdx + 224]
    vpsignb ymm7, ymm7, ymm6
    vpmaddubsw ymm8, ymm8, ymm7
    # sum of group 6
    vpmaddwd ymm6, ymm10, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 24]
    vfmadd231ps ymm4, ymm6, ymm7
    # scales of superblock +1
    vpbroadcastw xmm6, word ptr [rdi + 98]
    vcvtph2ps ymm6, xmm6
    vpor ymm5, ymm5, ymmword ptr [rip + {c9}]
    vcvtdq2ps ymm5, ymm5
    vmulps ymm5, ymm6, ymm5
    vmulps ymm5, ymm5, ymmword ptr [rip + {c10}]
    vmulps ymm5, ymm5, ymmword ptr [rcx + 32]
    vmovaps ymmword ptr [r9], ymm5
    # products of group 0 of superblock +1
    vpmovzxbd ymm5, qword ptr [rdi + 100]
    vpcmpeqd ymm6, ymm6, ymm6
    vxorps xmm7, xmm7, xmm7
    vpgatherdd ymm7, dword ptr [r8 + ymm5*4], ymm6
    vpbroadcastd ymm5, dword ptr [r9 + 32]
    vpshufb ymm5, ymm5, ymm0
    vpand ymm5, ymm5, ymm1
    vpaddb ymm5, ymm5, ymm2
    vmovdqa ymm6, ymmword ptr [rdx + 256]
    vpsignb ymm6, ymm6, ymm5
    vpmaddubsw ymm7, ymm7, ymm6
    # sum of group 7
    vpmaddwd ymm5, ymm8, ymm3
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 28]
    vfmadd231ps ymm4, ymm5, ymm6
    add rdi, 98
    add rdx, 256
    add rcx, 32
    xor rax, 64
    xor r9, 64
    dec rsi
    jnz 2b
3:
    # products of group 1 of superblock +0
    vpmovzxbd ymm5, qword ptr [rdi + 10]
    vpcmpeqd ymm6, ymm6, ymm6
    vxorps xmm11, xmm11, xmm11
    vpgatherdd ymm11, dword ptr [r8 + ymm5*4], ymm6
    vpbroadcastd ymm5, dword ptr [rax + 36]
    vpshufb ymm5, ymm5, ymm0
    vpand ymm5, ymm5, ymm1
    vpaddb ymm5, ymm5, ymm2
    vmovdqa ymm6, ymmword ptr [rdx + 32]
    vpsignb ymm6, ymm6, ymm5
    vpmaddubsw ymm11, ymm11, ymm6
    # sum of group 0
    vpmaddwd ymm5, ymm7, ymm3
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax]
    vfmadd231ps ymm4, ymm5, ymm6
    # products of group 2 of superblock +0
    vpmovzxbd ymm5, qword ptr [rdi + 18]
    vpcmpeqd ymm6, ymm6, ymm6
    vxorps xmm10, xmm10, xmm10
    vpgatherdd ymm10, dword ptr [r8 + ymm5*4], ymm6
    vpbroadcastd ymm5, dword ptr [rax + 40]
    vpshufb ymm5, ymm5, ymm0
    vpand ymm5, ymm5, ymm1
    vpaddb ymm5, ymm5, ymm2
    vmovdqa ymm6, ymmword ptr [rdx + 64]
    vpsignb ymm6, ymm6, ymm5
    vpmaddubsw ymm10, ymm10, ymm6
    # sum of group 1
    vpmaddwd ymm5, ymm11, ymm3
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 4]
    vfmadd231ps ymm4, ymm5, ymm6
    # products of group 3 of superblock +0
    vpmovzxbd ymm5, qword ptr [rdi + 26]
    vpcmpeqd ymm6, ymm6, ymm6
    vxorps xmm11, xmm11, xmm11
    vpgatherdd ymm11, dword ptr [r8 + ymm5*4], ymm6
    vpbroadcastd ymm5, dword ptr [rax + 44]
    vpshufb ymm5, ymm5, ymm0
    vpand ymm5, ymm5, ymm1
    vpaddb ymm5, ymm5, ymm2
    vmovdqa ymm6, ymmword ptr [rdx + 96]
    vpsignb ymm6, ymm6, ymm5
    vpmaddubsw ymm11, ymm11, ymm6
    # sum of group 2
    vpmaddwd ymm5, ymm10, ymm3
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 8]
    vfmadd231ps ymm4, ymm5, ymm6
    # products of group 4 of superblock +0
    vpmovzxbd ymm5, qword ptr [rdi + 34]
    vpcmpeqd ymm6, ymm6, ymm6
    vxorps xmm9, xmm9, xmm9
    vpgatherdd ymm9, dword ptr [r8 + ymm5*4], ymm6
    vpbroadcastd ymm5, dword ptr [rax + 48]
    vpshufb ymm5, ymm5, ymm0
    vpand ymm5, ymm5, ymm1
    vpaddb ymm5, ymm5, ymm2
    vmovdqa ymm6, ymmword ptr [rdx + 128]
    vpsignb ymm6, ymm6, ymm5
    vpmaddubsw ymm9, ymm9, ymm6
    # sum of group 3
    vpmaddwd ymm5, ymm11, ymm3
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 12]
    vfmadd231ps ymm4, ymm5, ymm6
    # products of group 5 of superblock +0
    vpmovzxbd ymm5, qword ptr [rdi + 42]
    vpcmpeqd ymm6, ymm6, ymm6
    vxorps xmm12, xmm12, xmm12
    vpgatherdd ymm12, dword ptr [r8 + ymm5*4], ymm6
    vpbroadcastd ymm5, dword ptr [rax + 52]
    vpshufb ymm5, ymm5, ymm0
    vpand ymm5, ymm5, ymm1
    vpaddb ymm5, ymm5, ymm2
    vmovdqa ymm6, ymmword ptr [rdx + 160]
    vpsignb ymm6, ymm6, ymm5
    vpmaddubsw ymm12, ymm12, ymm6
    # sum of group 4
    vpmaddwd ymm5, ymm9, ymm3
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 16]
    vfmadd231ps ymm4, ymm5, ymm6
    # products of group 6 of superblock +0
    vpmovzxbd ymm5, qword ptr [rdi + 50]
    vpcmpeqd ymm6, ymm6, ymm6
    vxorps xmm10, xmm10, xmm10
    vpgatherdd ymm10, dword ptr [r8 + ymm5*4], ymm6
    vpbroadcastd ymm5, dword ptr [rax + 56]
    vpshufb ymm5, ymm5, ymm0
    vpand ymm5, ymm5, ymm1
    vpaddb ymm5, ymm5, ymm2
    vmovdqa ymm6, ymmword ptr [rdx + 192]
    vpsignb ymm6, ymm6, ymm5
    vpmaddubsw ymm10, ymm10, ymm6
    # sum of group 5
    vpmaddwd ymm5, ymm12, ymm3
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 20]
    vfmadd231ps ymm4, ymm5, ymm6
    # products of group 7 of superblock +0
    vpmovzxbd ymm5, qword ptr [rdi + 58]
    vpcmpeqd ymm6, ymm6, ymm6
    vxorps xmm8, xmm8, xmm8
    vpgatherdd ymm8, dword ptr [r8 + ymm5*4], ymm6
    vpbroadcastd ymm5, dword ptr [rax + 60]
    vpshufb ymm5, ymm5, ymm0
    vpand ymm5, ymm5, ymm1
    vpaddb ymm5, ymm5, ymm2
    vmovdqa ymm0, ymmword ptr [rdx + 224]
    vpsignb ymm0, ymm0, ymm5
    vpmaddubsw ymm8, ymm8, ymm0
    # sum of group 6
    vpmaddwd ymm0, ymm10, ymm3
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm1, dword ptr [rax + 24]
    vfmadd231ps ymm4, ymm0, ymm1
    # sum of group 7
    vpmaddwd ymm0, ymm8, ymm3
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm1, dword ptr [rax + 28]
    vfmadd231ps ymm4, ymm0, ymm1
    # the lanes summed
    vxorps xmm0, xmm0, xmm0
    vaddps ymm4, ymm4, ymm0
    vextractf128 xmm0, ymm4, 1
    vaddps xmm0, xmm4, xmm0
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
#   00 00 00 00 00 00 00 00 01 01 01 01 01 01 01 01
#   02 02 02 02 02 02 02 02 03 03 03 03 03 03 03 03
# c1, aligned to 8:
#   01 02 04 08 10 20 40 80
# c2, aligned to 4:
#   7f 7f 7f 7f
# c3, aligned to 4:
#   01 00 01 00
# c4, aligned to 32:
#   7f 00 00 00 7f 00 00 00 7f 00 00 00 7f 00 00 00
#   7f 00 00 00 7f 00 00 00 7f 00 00 00 7f 00 00 00
# c5, aligned to 32:
#   00 7f 00 00 00 7f 00 00 00 7f 00 00 00 7f 00 00
#   00 7f 00 00 00 7f 00 00 00 7f 00 00 00 7f 00 00
# c6, aligned to 32:
#   00 00 7f 00 00 00 7f 00 00 00 7f 00 00 00 7f 00
#   00 00 7f 00 00 00 7f 00 00 00 7f 00 00 00 7f 00
# c7, aligned to 32:
#   00 00 00 7f 00 00 00 7f 00 00 00 7f 00 00 00 7f
#   00 00 00 7f 00 00 00 7f 00 00 00 7f 00 00 00 7f
# c8, aligned to 32:
#   80 80 80 80 80 80 80 80 80 80 80 80 80 80 80 80
#   80 80 80 80 80 80 80 80 80 80 80 80 80 80 80 80
# c9, aligned to 32:
#   01 00 00 00 01 00 00 00 01 00 00 00 01 00 00 00
#   01 00 00 00 01 00 00 00 01 00 00 00 01 00 00 00
# c10, aligned to 32:
#   00 00 80 3e 00 00 80 3e 00 00 80 3e 00 00 80 3e
#   00 00 80 3e 00 00 80 3e 00 00 80 3e 00 00 80 3e
