# iq3_s_ahead1(gpr, gpr, gpr, gpr, gpr) -> vec
# frame: 128 bytes aligned to 128, saves []
# row=rdi sbs=rsi quants=rdx scales=rcx grid=r8 spread=ymm0
# bit=ymm1 max8=ymm2 one16=ymm3 acc=ymm4 cur=rax nxt=r9
# p0=ymm7 p1=ymm9 p2=ymm7 p3=ymm8 p4=ymm7 p5=ymm8
# p6=ymm7 p7=ymm8 packed_signs=ymm5 d=ymm5 steps=ymm6 scale=ymm6
# indices=ymm5,ymm6 gather_mask=ymm6,ymm8 high_indices=ymm6,ymm8 signs=ymm5,ymm6 x=ymm6,ymm8,ymm0 lanes=ymm6,ymm5,ymm0
# group_scale=ymm7,ymm6,ymm1 zero=ymm0 high=ymm0 sum=ymm0 pair=ymm1 result=ymm0
# nothing=ymm0
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
    vmovdqu ymm5, ymmword ptr [rdi + 74]
    vmovdqa ymmword ptr [rax + 32], ymm5
    # scales of superblock +0
    vpbroadcastw xmm5, word ptr [rdi]
    vcvtph2ps ymm5, xmm5
    vmovd xmm6, dword ptr [rdi + 106]
    vpshufb xmm6, xmm6, xmmword ptr [rip + {c6}]
    vpmovzxbd ymm6, xmm6
    vpsrlvd ymm6, ymm6, ymmword ptr [rip + {c7}]
    vpand ymm6, ymm6, ymmword ptr [rip + {c8}]
    vpaddd ymm6, ymm6, ymm6
    vpor ymm6, ymm6, ymmword ptr [rip + {c9}]
    vcvtdq2ps ymm6, ymm6
    vmulps ymm6, ymm5, ymm6
    vmulps ymm6, ymm6, ymmword ptr [rcx]
    vmovaps ymmword ptr [rax], ymm6
    # products of group 0 of superblock +0
    vpmovzxbd ymm5, qword ptr [rdi + 2]
    vpbroadcastb ymm6, byte ptr [rdi + 66]
    vpand ymm6, ymm6, ymmword ptr [rip + {c4}]
    vpsllvd ymm6, ymm6, ymmword ptr [rip + {c5}]
    vpor ymm5, ymm5, ymm6
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
    vmovdqu ymm5, ymmword ptr [rdi + 184]
    # products of group 1 of superblock +0
    vpmovzxbd ymm6, qword ptr [rdi + 10]
    vpbroadcastb ymm8, byte ptr [rdi + 67]
    vpand ymm8, ymm8, ymmword ptr [rip + {c4}]
    vpsllvd ymm8, ymm8, ymmword ptr [rip + {c5}]
    vpor ymm6, ymm6, ymm8
    vpcmpeqd ymm8, ymm8, ymm8
    vxorps xmm9, xmm9, xmm9
    vpgatherdd ymm9, dword ptr [r8 + ymm6*4], ymm8
    vpbroadcastd ymm6, dword ptr [rax + 36]
    vpshufb ymm6, ymm6, ymm0
    vpand ymm6, ymm6, ymm1
    vpaddb ymm6, ymm6, ymm2
    vmovdqa ymm8, ymmword ptr [rdx + 32]
    vpsignb ymm8, ymm8, ymm6
    vpmaddubsw ymm9, ymm9, ymm8
    # sum of group 0
    vpmaddwd ymm6, ymm7, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax]
    vfmadd231ps ymm4, ymm6, ymm7
    vmovdqa ymmword ptr [r9 + 32], ymm5
    # products of group 2 of superblock +0
    vpmovzxbd ymm5, qword ptr [rdi + 18]
    vpbroadcastb ymm6, byte ptr [rdi + 68]
    vpand ymm6, ymm6, ymmword ptr [rip + {c4}]
    vpsllvd ymm6, ymm6, ymmword ptr [rip + {c5}]
    vpor ymm5, ymm5, ymm6
    vpcmpeqd ymm6, ymm6, ymm6
    vxorps xmm7, xmm7, xmm7
    vpgatherdd ymm7, dword ptr [r8 + ymm5*4], ymm6
    vpbroadcastd ymm5, dword ptr [rax + 40]
    vpshufb ymm5, ymm5, ymm0
    vpand ymm5, ymm5, ymm1
    vpaddb ymm5, ymm5, ymm2
    vmovdqa ymm6, ymmword ptr [rdx + 64]
    vpsignb ymm6, ymm6, ymm5
    vpmaddubsw ymm7, ymm7, ymm6
    # sum of group 1
    vpmaddwd ymm5, ymm9, ymm3
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 4]
    vfmadd231ps ymm4, ymm5, ymm6
    # products of group 3 of superblock +0
    vpmovzxbd ymm5, qword ptr [rdi + 26]
    vpbroadcastb ymm6, byte ptr [rdi + 69]
    vpand ymm6, ymm6, ymmword ptr [rip + {c4}]
    vpsllvd ymm6, ymm6, ymmword ptr [rip + {c5}]
    vpor ymm5, ymm5, ymm6
    vpcmpeqd ymm6, ymm6, ymm6
    vxorps xmm8, xmm8, xmm8
    vpgatherdd ymm8, dword ptr [r8 + ymm5*4], ymm6
    vpbroadcastd ymm5, dword ptr [rax + 44]
    vpshufb ymm5, ymm5, ymm0
    vpand ymm5, ymm5, ymm1
    vpaddb ymm5, ymm5, ymm2
    vmovdqa ymm6, ymmword ptr [rdx + 96]
    vpsignb ymm6, ymm6, ymm5
    vpmaddubsw ymm8, ymm8, ymm6
    # sum of group 2
    vpmaddwd ymm5, ymm7, ymm3
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 8]
    vfmadd231ps ymm4, ymm5, ymm6
    # products of group 4 of superblock +0
    vpmovzxbd ymm5, qword ptr [rdi + 34]
    vpbroadcastb ymm6, byte ptr [rdi + 70]
    vpand ymm6, ymm6, ymmword ptr [rip + {c4}]
    vpsllvd ymm6, ymm6, ymmword ptr [rip + {c5}]
    vpor ymm5, ymm5, ymm6
    vpcmpeqd ymm6, ymm6, ymm6
    vxorps xmm7, xmm7, xmm7
    vpgatherdd ymm7, dword ptr [r8 + ymm5*4], ymm6
    vpbroadcastd ymm5, dword ptr [rax + 48]
    vpshufb ymm5, ymm5, ymm0
    vpand ymm5, ymm5, ymm1
    vpaddb ymm5, ymm5, ymm2
    vmovdqa ymm6, ymmword ptr [rdx + 128]
    vpsignb ymm6, ymm6, ymm5
    vpmaddubsw ymm7, ymm7, ymm6
    # sum of group 3
    vpmaddwd ymm5, ymm8, ymm3
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 12]
    vfmadd231ps ymm4, ymm5, ymm6
    # products of group 5 of superblock +0
    vpmovzxbd ymm5, qword ptr [rdi + 42]
    vpbroadcastb ymm6, byte ptr [rdi + 71]
    vpand ymm6, ymm6, ymmword ptr [rip + {c4}]
    vpsllvd ymm6, ymm6, ymmword ptr [rip + {c5}]
    vpor ymm5, ymm5, ymm6
    vpcmpeqd ymm6, ymm6, ymm6
    vxorps xmm8, xmm8, xmm8
    vpgatherdd ymm8, dword ptr [r8 + ymm5*4], ymm6
    vpbroadcastd ymm5, dword ptr [rax + 52]
    vpshufb ymm5, ymm5, ymm0
    vpand ymm5, ymm5, ymm1
    vpaddb ymm5, ymm5, ymm2
    vmovdqa ymm6, ymmword ptr [rdx + 160]
    vpsignb ymm6, ymm6, ymm5
    vpmaddubsw ymm8, ymm8, ymm6
    # sum of group 4
    vpmaddwd ymm5, ymm7, ymm3
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 16]
    vfmadd231ps ymm4, ymm5, ymm6
    # products of group 6 of superblock +0
    vpmovzxbd ymm5, qword ptr [rdi + 50]
    vpbroadcastb ymm6, byte ptr [rdi + 72]
    vpand ymm6, ymm6, ymmword ptr [rip + {c4}]
    vpsllvd ymm6, ymm6, ymmword ptr [rip + {c5}]
    vpor ymm5, ymm5, ymm6
    vpcmpeqd ymm6, ymm6, ymm6
    vxorps xmm7, xmm7, xmm7
    vpgatherdd ymm7, dword ptr [r8 + ymm5*4], ymm6
    vpbroadcastd ymm5, dword ptr [rax + 56]
    vpshufb ymm5, ymm5, ymm0
    vpand ymm5, ymm5, ymm1
    vpaddb ymm5, ymm5, ymm2
    vmovdqa ymm6, ymmword ptr [rdx + 192]
    vpsignb ymm6, ymm6, ymm5
    vpmaddubsw ymm7, ymm7, ymm6
    # sum of group 5
    vpmaddwd ymm5, ymm8, ymm3
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 20]
    vfmadd231ps ymm4, ymm5, ymm6
    # products of group 7 of superblock +0
    vpmovzxbd ymm5, qword ptr [rdi + 58]
    vpbroadcastb ymm6, byte ptr [rdi + 73]
    vpand ymm6, ymm6, ymmword ptr [rip + {c4}]
    vpsllvd ymm6, ymm6, ymmword ptr [rip + {c5}]
    vpor ymm5, ymm5, ymm6
    vpcmpeqd ymm6, ymm6, ymm6
    vxorps xmm8, xmm8, xmm8
    vpgatherdd ymm8, dword ptr [r8 + ymm5*4], ymm6
    vpbroadcastd ymm5, dword ptr [rax + 60]
    vpshufb ymm5, ymm5, ymm0
    vpand ymm5, ymm5, ymm1
    vpaddb ymm5, ymm5, ymm2
    vmovdqa ymm6, ymmword ptr [rdx + 224]
    vpsignb ymm6, ymm6, ymm5
    vpmaddubsw ymm8, ymm8, ymm6
    # sum of group 6
    vpmaddwd ymm5, ymm7, ymm3
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 24]
    vfmadd231ps ymm4, ymm5, ymm6
    # scales of superblock +1
    vpbroadcastw xmm5, word ptr [rdi + 110]
    vcvtph2ps ymm5, xmm5
    vmovd xmm6, dword ptr [rdi + 216]
    vpshufb xmm6, xmm6, xmmword ptr [rip + {c6}]
    vpmovzxbd ymm6, xmm6
    vpsrlvd ymm6, ymm6, ymmword ptr [rip + {c7}]
    vpand ymm6, ymm6, ymmword ptr [rip + {c8}]
    vpaddd ymm6, ymm6, ymm6
    vpor ymm6, ymm6, ymmword ptr [rip + {c9}]
    vcvtdq2ps ymm6, ymm6
    vmulps ymm6, ymm5, ymm6
    vmulps ymm6, ymm6, ymmword ptr [rcx + 32]
    vmovaps ymmword ptr [r9], ymm6
    # products of group 0 of superblock +1
    vpmovzxbd ymm5, qword ptr [rdi + 112]
    vpbroadcastb ymm6, byte ptr [rdi + 176]
    vpand ymm6, ymm6, ymmword ptr [rip + {c4}]
    vpsllvd ymm6, ymm6, ymmword ptr [rip + {c5}]
    vpor ymm5, ymm5, ymm6
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
    add rdi, 110
    add rdx, 256
    add rcx, 32
    xor rax, 64
    xor r9, 64
    dec rsi
    jnz 2b
3:
    # products of group 1 of superblock +0
    vpmovzxbd ymm5, qword ptr [rdi + 10]
    vpbroadcastb ymm6, byte ptr [rdi + 67]
    vpand ymm6, ymm6, ymmword ptr [rip + {c4}]
    vpsllvd ymm6, ymm6, ymmword ptr [rip + {c5}]
    vpor ymm5, ymm5, ymm6
    vpcmpeqd ymm6, ymm6, ymm6
    vxorps xmm9, xmm9, xmm9
    vpgatherdd ymm9, dword ptr [r8 + ymm5*4], ymm6
    vpbroadcastd ymm5, dword ptr [rax + 36]
    vpshufb ymm5, ymm5, ymm0
    vpand ymm5, ymm5, ymm1
    vpaddb ymm5, ymm5, ymm2
    vmovdqa ymm6, ymmword ptr [rdx + 32]
    vpsignb ymm6, ymm6, ymm5
    vpmaddubsw ymm9, ymm9, ymm6
    # sum of group 0
    vpmaddwd ymm5, ymm7, ymm3
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax]
    vfmadd231ps ymm4, ymm5, ymm6
    # products of group 2 of superblock +0
    vpmovzxbd ymm5, qword ptr [rdi + 18]
    vpbroadcastb ymm6, byte ptr [rdi + 68]
    vpand ymm6, ymm6, ymmword ptr [rip + {c4}]
    vpsllvd ymm6, ymm6, ymmword ptr [rip + {c5}]
    vpor ymm5, ymm5, ymm6
    vpcmpeqd ymm6, ymm6, ymm6
    vxorps xmm7, xmm7, xmm7
    vpgatherdd ymm7, dword ptr [r8 + ymm5*4], ymm6
    vpbroadcastd ymm5, dword ptr [rax + 40]
    vpshufb ymm5, ymm5, ymm0
    vpand ymm5, ymm5, ymm1
    vpaddb ymm5, ymm5, ymm2
    vmovdqa ymm6, ymmword ptr [rdx + 64]
    vpsignb ymm6, ymm6, ymm5
    vpmaddubsw ymm7, ymm7, ymm6
    # sum of group 1
    vpmaddwd ymm5, ymm9, ymm3
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 4]
    vfmadd231ps ymm4, ymm5, ymm6
    # products of group 3 of superblock +0
    vpmovzxbd ymm5, qword ptr [rdi + 26]
    vpbroadcastb ymm6, byte ptr [rdi + 69]
    vpand ymm6, ymm6, ymmword ptr [rip + {c4}]
    vpsllvd ymm6, ymm6, ymmword ptr [rip + {c5}]
    vpor ymm5, ymm5, ymm6
    vpcmpeqd ymm6, ymm6, ymm6
    vxorps xmm8, xmm8, xmm8
    vpgatherdd ymm8, dword ptr [r8 + ymm5*4], ymm6
    vpbroadcastd ymm5, dword ptr [rax + 44]
    vpshufb ymm5, ymm5, ymm0
    vpand ymm5, ymm5, ymm1
    vpaddb ymm5, ymm5, ymm2
    vmovdqa ymm6, ymmword ptr [rdx + 96]
    vpsignb ymm6, ymm6, ymm5
    vpmaddubsw ymm8, ymm8, ymm6
    # sum of group 2
    vpmaddwd ymm5, ymm7, ymm3
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 8]
    vfmadd231ps ymm4, ymm5, ymm6
    # products of group 4 of superblock +0
    vpmovzxbd ymm5, qword ptr [rdi + 34]
    vpbroadcastb ymm6, byte ptr [rdi + 70]
    vpand ymm6, ymm6, ymmword ptr [rip + {c4}]
    vpsllvd ymm6, ymm6, ymmword ptr [rip + {c5}]
    vpor ymm5, ymm5, ymm6
    vpcmpeqd ymm6, ymm6, ymm6
    vxorps xmm7, xmm7, xmm7
    vpgatherdd ymm7, dword ptr [r8 + ymm5*4], ymm6
    vpbroadcastd ymm5, dword ptr [rax + 48]
    vpshufb ymm5, ymm5, ymm0
    vpand ymm5, ymm5, ymm1
    vpaddb ymm5, ymm5, ymm2
    vmovdqa ymm6, ymmword ptr [rdx + 128]
    vpsignb ymm6, ymm6, ymm5
    vpmaddubsw ymm7, ymm7, ymm6
    # sum of group 3
    vpmaddwd ymm5, ymm8, ymm3
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 12]
    vfmadd231ps ymm4, ymm5, ymm6
    # products of group 5 of superblock +0
    vpmovzxbd ymm5, qword ptr [rdi + 42]
    vpbroadcastb ymm6, byte ptr [rdi + 71]
    vpand ymm6, ymm6, ymmword ptr [rip + {c4}]
    vpsllvd ymm6, ymm6, ymmword ptr [rip + {c5}]
    vpor ymm5, ymm5, ymm6
    vpcmpeqd ymm6, ymm6, ymm6
    vxorps xmm8, xmm8, xmm8
    vpgatherdd ymm8, dword ptr [r8 + ymm5*4], ymm6
    vpbroadcastd ymm5, dword ptr [rax + 52]
    vpshufb ymm5, ymm5, ymm0
    vpand ymm5, ymm5, ymm1
    vpaddb ymm5, ymm5, ymm2
    vmovdqa ymm6, ymmword ptr [rdx + 160]
    vpsignb ymm6, ymm6, ymm5
    vpmaddubsw ymm8, ymm8, ymm6
    # sum of group 4
    vpmaddwd ymm5, ymm7, ymm3
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 16]
    vfmadd231ps ymm4, ymm5, ymm6
    # products of group 6 of superblock +0
    vpmovzxbd ymm5, qword ptr [rdi + 50]
    vpbroadcastb ymm6, byte ptr [rdi + 72]
    vpand ymm6, ymm6, ymmword ptr [rip + {c4}]
    vpsllvd ymm6, ymm6, ymmword ptr [rip + {c5}]
    vpor ymm5, ymm5, ymm6
    vpcmpeqd ymm6, ymm6, ymm6
    vxorps xmm7, xmm7, xmm7
    vpgatherdd ymm7, dword ptr [r8 + ymm5*4], ymm6
    vpbroadcastd ymm5, dword ptr [rax + 56]
    vpshufb ymm5, ymm5, ymm0
    vpand ymm5, ymm5, ymm1
    vpaddb ymm5, ymm5, ymm2
    vmovdqa ymm6, ymmword ptr [rdx + 192]
    vpsignb ymm6, ymm6, ymm5
    vpmaddubsw ymm7, ymm7, ymm6
    # sum of group 5
    vpmaddwd ymm5, ymm8, ymm3
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 20]
    vfmadd231ps ymm4, ymm5, ymm6
    # products of group 7 of superblock +0
    vpmovzxbd ymm5, qword ptr [rdi + 58]
    vpbroadcastb ymm6, byte ptr [rdi + 73]
    vpand ymm6, ymm6, ymmword ptr [rip + {c4}]
    vpsllvd ymm6, ymm6, ymmword ptr [rip + {c5}]
    vpor ymm5, ymm5, ymm6
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
    vpmaddwd ymm0, ymm7, ymm3
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
#   01 00 00 00 02 00 00 00 04 00 00 00 08 00 00 00
#   10 00 00 00 20 00 00 00 40 00 00 00 80 00 00 00
# c5, aligned to 32:
#   08 00 00 00 07 00 00 00 06 00 00 00 05 00 00 00
#   04 00 00 00 03 00 00 00 02 00 00 00 01 00 00 00
# c6, aligned to 16:
#   00 00 01 01 02 02 03 03 80 80 80 80 80 80 80 80
# c7, aligned to 32:
#   00 00 00 00 04 00 00 00 00 00 00 00 04 00 00 00
#   00 00 00 00 04 00 00 00 00 00 00 00 04 00 00 00
# c8, aligned to 32:
#   0f 00 00 00 0f 00 00 00 0f 00 00 00 0f 00 00 00
#   0f 00 00 00 0f 00 00 00 0f 00 00 00 0f 00 00 00
# c9, aligned to 32:
#   01 00 00 00 01 00 00 00 01 00 00 00 01 00 00 00
#   01 00 00 00 01 00 00 00 01 00 00 00 01 00 00 00
