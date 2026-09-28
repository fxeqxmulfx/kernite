# iq3_xxs2_ahead1(gpr, gpr, gpr, gpr, gpr, gpr) -> ()
# frame: 256 bytes aligned to 256, saves []
# row=rdi sbs=rsi quants=rdx scales=rcx grid=r8 out=r9
# spread=ymm0 bit=ymm1 max8=ymm2 one16=ymm3 acc0=ymm4 acc1=ymm5
# cur=rax nxt=r10 p0a=ymm7 p0b=ymm6 p1a=ymm10 p1b=ymm9
# p2a=ymm10 p2b=ymm9 p3a=ymm10 p3b=ymm9 p4a=ymm9 p4b=ymm8
# p5a=ymm11 p5b=ymm10 p6a=ymm10 p6b=ymm8 p7a=ymm8 p7b=ymm7
# steps=ymm6 words=ymm6 fields=ymm7 field=ymm8 parity=ymm9 d=ymm7
# scale=ymm6 pair_scale=ymm7,ymm6 indices=ymm6,ymm9,ymm8,ymm10,ymm7 gather_mask=ymm7,ymm10,ymm9,ymm11,ymm8 levels=ymm8,ymm11,ymm10,ymm12,ymm9 signs=ymm6,ymm9,ymm8,ymm10,ymm7
# x=ymm7,ymm9,ymm10,ymm12,ymm11,ymm13,ymm8,ymm0 lanes=ymm7,ymm6,ymm10,ymm9,ymm8,ymm11,ymm0 group_scale=ymm8,ymm7,ymm11,ymm10,ymm9,ymm12,ymm1 zero=ymm0 high=ymm0 sum=ymm0
# pair=ymm1 result=ymm0 nothing=ymm0
    push rbp
    mov rbp, rsp
    and rsp, -256
    sub rsp, 256
    test rsi, rsi
    jz 4f
    # constants
    vmovdqa ymm0, ymmword ptr [rip + {c0}]
    vpbroadcastq ymm1, qword ptr [rip + {c1}]
    vpbroadcastd ymm2, dword ptr [rip + {c2}]
    vpbroadcastd ymm3, dword ptr [rip + {c3}]
    vxorps xmm4, xmm4, xmm4
    vxorps xmm5, xmm5, xmm5
    lea rax, [rsp]
    lea r10, [rsp + 128]
    # signs of superblock +0
    vmovdqu ymm6, ymmword ptr [rdi + 66]
    vpand ymm7, ymm6, ymmword ptr [rip + {c4}]
    vpaddd ymm8, ymm6, ymm6
    vpand ymm8, ymm8, ymmword ptr [rip + {c5}]
    vpor ymm7, ymm7, ymm8
    vpslld ymm8, ymm6, 2
    vpand ymm8, ymm8, ymmword ptr [rip + {c6}]
    vpor ymm7, ymm7, ymm8
    vpslld ymm8, ymm6, 3
    vpand ymm8, ymm8, ymmword ptr [rip + {c7}]
    vpor ymm7, ymm7, ymm8
    vpsrld ymm6, ymm6, 27
    vpaddw ymm9, ymm7, ymm7
    vpxor ymm9, ymm9, ymm7
    vpsllw ymm8, ymm9, 2
    vpxor ymm9, ymm9, ymm8
    vpsllw ymm8, ymm9, 4
    vpxor ymm9, ymm9, ymm8
    vpand ymm9, ymm9, ymmword ptr [rip + {c8}]
    vpor ymm7, ymm7, ymm9
    vmovdqa ymmword ptr [rax + 64], ymm7
    # pair scales of superblock +0
    vpbroadcastw xmm7, word ptr [rdi]
    vcvtph2ps ymm7, xmm7
    vpor ymm6, ymm6, ymmword ptr [rip + {c9}]
    vcvtdq2ps ymm6, ymm6
    vmulps ymm6, ymm7, ymm6
    vmulps ymm6, ymm6, ymmword ptr [rip + {c10}]
    vpermq ymm6, ymm6, 0xd8
    vshufps ymm7, ymm6, ymm6, 0x50
    vmulps ymm7, ymm7, ymmword ptr [rcx]
    vmovaps ymmword ptr [rax], ymm7
    vshufps ymm6, ymm6, ymm6, 0xfa
    vmulps ymm6, ymm6, ymmword ptr [rcx + 32]
    vmovaps ymmword ptr [rax + 32], ymm6
    # products of group 0 of superblock +0
    vpmovzxbd ymm6, qword ptr [rdi + 2]
    vpcmpeqd ymm7, ymm7, ymm7
    vxorps xmm8, xmm8, xmm8
    vpgatherdd ymm8, dword ptr [r8 + ymm6*4], ymm7
    vpbroadcastd ymm6, dword ptr [rax + 64]
    vpshufb ymm6, ymm6, ymm0
    vpand ymm6, ymm6, ymm1
    vpaddb ymm6, ymm6, ymm2
    vmovdqa ymm7, ymmword ptr [rdx]
    vpsignb ymm7, ymm7, ymm6
    vpmaddubsw ymm7, ymm8, ymm7
    vmovdqa ymm9, ymmword ptr [rdx + 32]
    vpsignb ymm9, ymm9, ymm6
    vpmaddubsw ymm6, ymm8, ymm9
    dec rsi
    jz 3f
2:
    # sums of group 0
    vpmaddwd ymm7, ymm7, ymm3
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rax]
    vfmadd231ps ymm4, ymm7, ymm8
    vpmaddwd ymm6, ymm6, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 4]
    vfmadd231ps ymm5, ymm6, ymm7
    # signs of superblock +1
    vmovdqu ymm6, ymmword ptr [rdi + 164]
    vpand ymm7, ymm6, ymmword ptr [rip + {c4}]
    vpaddd ymm8, ymm6, ymm6
    # products of group 1 of superblock +0
    vpmovzxbd ymm9, qword ptr [rdi + 10]
    vpcmpeqd ymm10, ymm10, ymm10
    vxorps xmm11, xmm11, xmm11
    vpgatherdd ymm11, dword ptr [r8 + ymm9*4], ymm10
    vpbroadcastd ymm9, dword ptr [rax + 68]
    vpshufb ymm9, ymm9, ymm0
    vpand ymm9, ymm9, ymm1
    vpaddb ymm9, ymm9, ymm2
    vmovdqa ymm10, ymmword ptr [rdx + 64]
    vpsignb ymm10, ymm10, ymm9
    vpmaddubsw ymm10, ymm11, ymm10
    vmovdqa ymm12, ymmword ptr [rdx + 96]
    vpsignb ymm12, ymm12, ymm9
    vpmaddubsw ymm9, ymm11, ymm12
    # sums of group 1
    vpmaddwd ymm10, ymm10, ymm3
    vcvtdq2ps ymm10, ymm10
    vbroadcastss ymm11, dword ptr [rax + 8]
    vfmadd231ps ymm4, ymm10, ymm11
    vpmaddwd ymm9, ymm9, ymm3
    vcvtdq2ps ymm9, ymm9
    vbroadcastss ymm10, dword ptr [rax + 12]
    vfmadd231ps ymm5, ymm9, ymm10
    vpand ymm8, ymm8, ymmword ptr [rip + {c5}]
    vpor ymm7, ymm7, ymm8
    vpslld ymm8, ymm6, 2
    # products of group 2 of superblock +0
    vpmovzxbd ymm9, qword ptr [rdi + 18]
    vpcmpeqd ymm10, ymm10, ymm10
    vxorps xmm11, xmm11, xmm11
    vpgatherdd ymm11, dword ptr [r8 + ymm9*4], ymm10
    vpbroadcastd ymm9, dword ptr [rax + 72]
    vpshufb ymm9, ymm9, ymm0
    vpand ymm9, ymm9, ymm1
    vpaddb ymm9, ymm9, ymm2
    vmovdqa ymm10, ymmword ptr [rdx + 128]
    vpsignb ymm10, ymm10, ymm9
    vpmaddubsw ymm10, ymm11, ymm10
    vmovdqa ymm12, ymmword ptr [rdx + 160]
    vpsignb ymm12, ymm12, ymm9
    vpmaddubsw ymm9, ymm11, ymm12
    # sums of group 2
    vpmaddwd ymm10, ymm10, ymm3
    vcvtdq2ps ymm10, ymm10
    vbroadcastss ymm11, dword ptr [rax + 16]
    vfmadd231ps ymm4, ymm10, ymm11
    vpmaddwd ymm9, ymm9, ymm3
    vcvtdq2ps ymm9, ymm9
    vbroadcastss ymm10, dword ptr [rax + 20]
    vfmadd231ps ymm5, ymm9, ymm10
    vpand ymm8, ymm8, ymmword ptr [rip + {c6}]
    vpor ymm7, ymm7, ymm8
    vpslld ymm8, ymm6, 3
    # products of group 3 of superblock +0
    vpmovzxbd ymm9, qword ptr [rdi + 26]
    vpcmpeqd ymm10, ymm10, ymm10
    vxorps xmm11, xmm11, xmm11
    vpgatherdd ymm11, dword ptr [r8 + ymm9*4], ymm10
    vpbroadcastd ymm9, dword ptr [rax + 76]
    vpshufb ymm9, ymm9, ymm0
    vpand ymm9, ymm9, ymm1
    vpaddb ymm9, ymm9, ymm2
    vmovdqa ymm10, ymmword ptr [rdx + 192]
    vpsignb ymm10, ymm10, ymm9
    vpmaddubsw ymm10, ymm11, ymm10
    vmovdqa ymm12, ymmword ptr [rdx + 224]
    vpsignb ymm12, ymm12, ymm9
    vpmaddubsw ymm9, ymm11, ymm12
    # sums of group 3
    vpmaddwd ymm10, ymm10, ymm3
    vcvtdq2ps ymm10, ymm10
    vbroadcastss ymm11, dword ptr [rax + 24]
    vfmadd231ps ymm4, ymm10, ymm11
    vpmaddwd ymm9, ymm9, ymm3
    vcvtdq2ps ymm9, ymm9
    vbroadcastss ymm10, dword ptr [rax + 28]
    vfmadd231ps ymm5, ymm9, ymm10
    vpand ymm8, ymm8, ymmword ptr [rip + {c7}]
    vpor ymm7, ymm7, ymm8
    vpsrld ymm6, ymm6, 27
    # products of group 4 of superblock +0
    vpmovzxbd ymm8, qword ptr [rdi + 34]
    vpcmpeqd ymm9, ymm9, ymm9
    vxorps xmm10, xmm10, xmm10
    vpgatherdd ymm10, dword ptr [r8 + ymm8*4], ymm9
    vpbroadcastd ymm8, dword ptr [rax + 80]
    vpshufb ymm8, ymm8, ymm0
    vpand ymm8, ymm8, ymm1
    vpaddb ymm8, ymm8, ymm2
    vmovdqa ymm9, ymmword ptr [rdx + 256]
    vpsignb ymm9, ymm9, ymm8
    vpmaddubsw ymm9, ymm10, ymm9
    vmovdqa ymm11, ymmword ptr [rdx + 288]
    vpsignb ymm11, ymm11, ymm8
    vpmaddubsw ymm8, ymm10, ymm11
    # sums of group 4
    vpmaddwd ymm9, ymm9, ymm3
    vcvtdq2ps ymm9, ymm9
    vbroadcastss ymm10, dword ptr [rax + 32]
    vfmadd231ps ymm4, ymm9, ymm10
    vpmaddwd ymm8, ymm8, ymm3
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm9, dword ptr [rax + 36]
    vfmadd231ps ymm5, ymm8, ymm9
    vpaddw ymm9, ymm7, ymm7
    vpxor ymm9, ymm9, ymm7
    vpsllw ymm8, ymm9, 2
    # products of group 5 of superblock +0
    vpmovzxbd ymm10, qword ptr [rdi + 42]
    vpcmpeqd ymm11, ymm11, ymm11
    vxorps xmm12, xmm12, xmm12
    vpgatherdd ymm12, dword ptr [r8 + ymm10*4], ymm11
    vpbroadcastd ymm10, dword ptr [rax + 84]
    vpshufb ymm10, ymm10, ymm0
    vpand ymm10, ymm10, ymm1
    vpaddb ymm10, ymm10, ymm2
    vmovdqa ymm11, ymmword ptr [rdx + 320]
    vpsignb ymm11, ymm11, ymm10
    vpmaddubsw ymm11, ymm12, ymm11
    vmovdqa ymm13, ymmword ptr [rdx + 352]
    vpsignb ymm13, ymm13, ymm10
    vpmaddubsw ymm10, ymm12, ymm13
    # sums of group 5
    vpmaddwd ymm11, ymm11, ymm3
    vcvtdq2ps ymm11, ymm11
    vbroadcastss ymm12, dword ptr [rax + 40]
    vfmadd231ps ymm4, ymm11, ymm12
    vpmaddwd ymm10, ymm10, ymm3
    vcvtdq2ps ymm10, ymm10
    vbroadcastss ymm11, dword ptr [rax + 44]
    vfmadd231ps ymm5, ymm10, ymm11
    vpxor ymm9, ymm9, ymm8
    vpsllw ymm8, ymm9, 4
    vpxor ymm9, ymm9, ymm8
    # products of group 6 of superblock +0
    vpmovzxbd ymm8, qword ptr [rdi + 50]
    vpcmpeqd ymm10, ymm10, ymm10
    vxorps xmm11, xmm11, xmm11
    vpgatherdd ymm11, dword ptr [r8 + ymm8*4], ymm10
    vpbroadcastd ymm8, dword ptr [rax + 88]
    vpshufb ymm8, ymm8, ymm0
    vpand ymm8, ymm8, ymm1
    vpaddb ymm8, ymm8, ymm2
    vmovdqa ymm10, ymmword ptr [rdx + 384]
    vpsignb ymm10, ymm10, ymm8
    vpmaddubsw ymm10, ymm11, ymm10
    vmovdqa ymm12, ymmword ptr [rdx + 416]
    vpsignb ymm12, ymm12, ymm8
    vpmaddubsw ymm8, ymm11, ymm12
    # sums of group 6
    vpmaddwd ymm10, ymm10, ymm3
    vcvtdq2ps ymm10, ymm10
    vbroadcastss ymm11, dword ptr [rax + 48]
    vfmadd231ps ymm4, ymm10, ymm11
    vpmaddwd ymm8, ymm8, ymm3
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm10, dword ptr [rax + 52]
    vfmadd231ps ymm5, ymm8, ymm10
    vpand ymm9, ymm9, ymmword ptr [rip + {c8}]
    vpor ymm7, ymm7, ymm9
    vmovdqa ymmword ptr [r10 + 64], ymm7
    # products of group 7 of superblock +0
    vpmovzxbd ymm7, qword ptr [rdi + 58]
    vpcmpeqd ymm8, ymm8, ymm8
    vxorps xmm9, xmm9, xmm9
    vpgatherdd ymm9, dword ptr [r8 + ymm7*4], ymm8
    vpbroadcastd ymm7, dword ptr [rax + 92]
    vpshufb ymm7, ymm7, ymm0
    vpand ymm7, ymm7, ymm1
    vpaddb ymm7, ymm7, ymm2
    vmovdqa ymm8, ymmword ptr [rdx + 448]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm8, ymm9, ymm8
    vmovdqa ymm10, ymmword ptr [rdx + 480]
    vpsignb ymm10, ymm10, ymm7
    vpmaddubsw ymm7, ymm9, ymm10
    # sums of group 7
    vpmaddwd ymm8, ymm8, ymm3
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm9, dword ptr [rax + 56]
    vfmadd231ps ymm4, ymm8, ymm9
    vpmaddwd ymm7, ymm7, ymm3
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rax + 60]
    vfmadd231ps ymm5, ymm7, ymm8
    # pair scales of superblock +1
    vpbroadcastw xmm7, word ptr [rdi + 98]
    vcvtph2ps ymm7, xmm7
    vpor ymm6, ymm6, ymmword ptr [rip + {c9}]
    vcvtdq2ps ymm6, ymm6
    vmulps ymm6, ymm7, ymm6
    vmulps ymm6, ymm6, ymmword ptr [rip + {c10}]
    vpermq ymm6, ymm6, 0xd8
    vshufps ymm7, ymm6, ymm6, 0x50
    vmulps ymm7, ymm7, ymmword ptr [rcx + 64]
    vmovaps ymmword ptr [r10], ymm7
    vshufps ymm6, ymm6, ymm6, 0xfa
    vmulps ymm6, ymm6, ymmword ptr [rcx + 96]
    vmovaps ymmword ptr [r10 + 32], ymm6
    # products of group 0 of superblock +1
    vpmovzxbd ymm6, qword ptr [rdi + 100]
    vpcmpeqd ymm7, ymm7, ymm7
    vxorps xmm8, xmm8, xmm8
    vpgatherdd ymm8, dword ptr [r8 + ymm6*4], ymm7
    vpbroadcastd ymm6, dword ptr [r10 + 64]
    vpshufb ymm6, ymm6, ymm0
    vpand ymm6, ymm6, ymm1
    vpaddb ymm6, ymm6, ymm2
    vmovdqa ymm7, ymmword ptr [rdx + 512]
    vpsignb ymm7, ymm7, ymm6
    vpmaddubsw ymm7, ymm8, ymm7
    vmovdqa ymm9, ymmword ptr [rdx + 544]
    vpsignb ymm9, ymm9, ymm6
    vpmaddubsw ymm6, ymm8, ymm9
    add rdi, 98
    add rdx, 512
    add rcx, 64
    xor rax, 128
    xor r10, 128
    dec rsi
    jnz 2b
3:
    # sums of group 0
    vpmaddwd ymm7, ymm7, ymm3
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rax]
    vfmadd231ps ymm4, ymm7, ymm8
    vpmaddwd ymm6, ymm6, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 4]
    vfmadd231ps ymm5, ymm6, ymm7
    # products of group 1 of superblock +0
    vpmovzxbd ymm6, qword ptr [rdi + 10]
    vpcmpeqd ymm7, ymm7, ymm7
    vxorps xmm8, xmm8, xmm8
    vpgatherdd ymm8, dword ptr [r8 + ymm6*4], ymm7
    vpbroadcastd ymm6, dword ptr [rax + 68]
    vpshufb ymm6, ymm6, ymm0
    vpand ymm6, ymm6, ymm1
    vpaddb ymm6, ymm6, ymm2
    vmovdqa ymm7, ymmword ptr [rdx + 64]
    vpsignb ymm7, ymm7, ymm6
    vpmaddubsw ymm10, ymm8, ymm7
    vmovdqa ymm7, ymmword ptr [rdx + 96]
    vpsignb ymm7, ymm7, ymm6
    vpmaddubsw ymm9, ymm8, ymm7
    # sums of group 1
    vpmaddwd ymm6, ymm10, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 8]
    vfmadd231ps ymm4, ymm6, ymm7
    vpmaddwd ymm6, ymm9, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 12]
    vfmadd231ps ymm5, ymm6, ymm7
    # products of group 2 of superblock +0
    vpmovzxbd ymm6, qword ptr [rdi + 18]
    vpcmpeqd ymm7, ymm7, ymm7
    vxorps xmm8, xmm8, xmm8
    vpgatherdd ymm8, dword ptr [r8 + ymm6*4], ymm7
    vpbroadcastd ymm6, dword ptr [rax + 72]
    vpshufb ymm6, ymm6, ymm0
    vpand ymm6, ymm6, ymm1
    vpaddb ymm6, ymm6, ymm2
    vmovdqa ymm7, ymmword ptr [rdx + 128]
    vpsignb ymm7, ymm7, ymm6
    vpmaddubsw ymm10, ymm8, ymm7
    vmovdqa ymm7, ymmword ptr [rdx + 160]
    vpsignb ymm7, ymm7, ymm6
    vpmaddubsw ymm9, ymm8, ymm7
    # sums of group 2
    vpmaddwd ymm6, ymm10, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 16]
    vfmadd231ps ymm4, ymm6, ymm7
    vpmaddwd ymm6, ymm9, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 20]
    vfmadd231ps ymm5, ymm6, ymm7
    # products of group 3 of superblock +0
    vpmovzxbd ymm6, qword ptr [rdi + 26]
    vpcmpeqd ymm7, ymm7, ymm7
    vxorps xmm8, xmm8, xmm8
    vpgatherdd ymm8, dword ptr [r8 + ymm6*4], ymm7
    vpbroadcastd ymm6, dword ptr [rax + 76]
    vpshufb ymm6, ymm6, ymm0
    vpand ymm6, ymm6, ymm1
    vpaddb ymm6, ymm6, ymm2
    vmovdqa ymm7, ymmword ptr [rdx + 192]
    vpsignb ymm7, ymm7, ymm6
    vpmaddubsw ymm10, ymm8, ymm7
    vmovdqa ymm7, ymmword ptr [rdx + 224]
    vpsignb ymm7, ymm7, ymm6
    vpmaddubsw ymm9, ymm8, ymm7
    # sums of group 3
    vpmaddwd ymm6, ymm10, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 24]
    vfmadd231ps ymm4, ymm6, ymm7
    vpmaddwd ymm6, ymm9, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 28]
    vfmadd231ps ymm5, ymm6, ymm7
    # products of group 4 of superblock +0
    vpmovzxbd ymm6, qword ptr [rdi + 34]
    vpcmpeqd ymm7, ymm7, ymm7
    vxorps xmm8, xmm8, xmm8
    vpgatherdd ymm8, dword ptr [r8 + ymm6*4], ymm7
    vpbroadcastd ymm6, dword ptr [rax + 80]
    vpshufb ymm6, ymm6, ymm0
    vpand ymm6, ymm6, ymm1
    vpaddb ymm6, ymm6, ymm2
    vmovdqa ymm7, ymmword ptr [rdx + 256]
    vpsignb ymm7, ymm7, ymm6
    vpmaddubsw ymm9, ymm8, ymm7
    vmovdqa ymm7, ymmword ptr [rdx + 288]
    vpsignb ymm7, ymm7, ymm6
    vpmaddubsw ymm8, ymm8, ymm7
    # sums of group 4
    vpmaddwd ymm6, ymm9, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 32]
    vfmadd231ps ymm4, ymm6, ymm7
    vpmaddwd ymm6, ymm8, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 36]
    vfmadd231ps ymm5, ymm6, ymm7
    # products of group 5 of superblock +0
    vpmovzxbd ymm6, qword ptr [rdi + 42]
    vpcmpeqd ymm7, ymm7, ymm7
    vxorps xmm8, xmm8, xmm8
    vpgatherdd ymm8, dword ptr [r8 + ymm6*4], ymm7
    vpbroadcastd ymm6, dword ptr [rax + 84]
    vpshufb ymm6, ymm6, ymm0
    vpand ymm6, ymm6, ymm1
    vpaddb ymm6, ymm6, ymm2
    vmovdqa ymm7, ymmword ptr [rdx + 320]
    vpsignb ymm7, ymm7, ymm6
    vpmaddubsw ymm11, ymm8, ymm7
    vmovdqa ymm7, ymmword ptr [rdx + 352]
    vpsignb ymm7, ymm7, ymm6
    vpmaddubsw ymm10, ymm8, ymm7
    # sums of group 5
    vpmaddwd ymm6, ymm11, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 40]
    vfmadd231ps ymm4, ymm6, ymm7
    vpmaddwd ymm6, ymm10, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 44]
    vfmadd231ps ymm5, ymm6, ymm7
    # products of group 6 of superblock +0
    vpmovzxbd ymm6, qword ptr [rdi + 50]
    vpcmpeqd ymm7, ymm7, ymm7
    vxorps xmm8, xmm8, xmm8
    vpgatherdd ymm8, dword ptr [r8 + ymm6*4], ymm7
    vpbroadcastd ymm6, dword ptr [rax + 88]
    vpshufb ymm6, ymm6, ymm0
    vpand ymm6, ymm6, ymm1
    vpaddb ymm6, ymm6, ymm2
    vmovdqa ymm7, ymmword ptr [rdx + 384]
    vpsignb ymm7, ymm7, ymm6
    vpmaddubsw ymm10, ymm8, ymm7
    vmovdqa ymm7, ymmword ptr [rdx + 416]
    vpsignb ymm7, ymm7, ymm6
    vpmaddubsw ymm8, ymm8, ymm7
    # sums of group 6
    vpmaddwd ymm6, ymm10, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 48]
    vfmadd231ps ymm4, ymm6, ymm7
    vpmaddwd ymm6, ymm8, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 52]
    vfmadd231ps ymm5, ymm6, ymm7
    # products of group 7 of superblock +0
    vpmovzxbd ymm6, qword ptr [rdi + 58]
    vpcmpeqd ymm7, ymm7, ymm7
    vxorps xmm9, xmm9, xmm9
    vpgatherdd ymm9, dword ptr [r8 + ymm6*4], ymm7
    vpbroadcastd ymm6, dword ptr [rax + 92]
    vpshufb ymm6, ymm6, ymm0
    vpand ymm6, ymm6, ymm1
    vpaddb ymm6, ymm6, ymm2
    vmovdqa ymm0, ymmword ptr [rdx + 448]
    vpsignb ymm0, ymm0, ymm6
    vpmaddubsw ymm8, ymm9, ymm0
    vmovdqa ymm0, ymmword ptr [rdx + 480]
    vpsignb ymm0, ymm0, ymm6
    vpmaddubsw ymm7, ymm9, ymm0
    # sums of group 7
    vpmaddwd ymm0, ymm8, ymm3
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm1, dword ptr [rax + 56]
    vfmadd231ps ymm4, ymm0, ymm1
    vpmaddwd ymm0, ymm7, ymm3
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm1, dword ptr [rax + 60]
    vfmadd231ps ymm5, ymm0, ymm1
    # the lanes summed
    vxorps xmm0, xmm0, xmm0
    vaddps ymm4, ymm4, ymm0
    vextractf128 xmm0, ymm4, 1
    vaddps xmm0, xmm4, xmm0
    vmovhlps xmm1, xmm0, xmm0
    vaddps xmm0, xmm0, xmm1
    vmovshdup xmm1, xmm0
    vaddss xmm0, xmm0, xmm1
    vmovss dword ptr [r9], xmm0
    # the lanes summed
    vxorps xmm0, xmm0, xmm0
    vaddps ymm5, ymm5, ymm0
    vextractf128 xmm0, ymm5, 1
    vaddps xmm0, xmm5, xmm0
    vmovhlps xmm1, xmm0, xmm0
    vaddps xmm0, xmm0, xmm1
    vmovshdup xmm1, xmm0
    vaddss xmm0, xmm0, xmm1
    vmovss dword ptr [r9 + 4], xmm0
    mov rsp, rbp
    pop rbp
    vzeroupper
    ret
4:
    vxorps xmm0, xmm0, xmm0
    vmovss dword ptr [r9], xmm0
    vmovss dword ptr [r9 + 4], xmm0
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
