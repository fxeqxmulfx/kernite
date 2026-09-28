# iq3_s_nibble2_ahead3(gpr, gpr, gpr, gpr, gpr) -> ()
# frame: 128 bytes aligned to 128, saves []
# row=rdi sbs=rsi quants=rdx scales=rcx out=r8 levels=ymm0
# nibble=ymm1 one16=ymm2 acc0=ymm3 acc1=ymm4 cur=rax nxt=r9
# p0a=ymm7 p0b=ymm5 p1a=ymm9 p1b=ymm6 p2a=ymm11 p2b=ymm8
# p3a=ymm13 p3b=ymm10 p4a=ymm12 p4b=ymm6 p5a=ymm11 p5b=ymm8
# p6a=ymm13 p6b=ymm9 p7a=ymm10 p7b=ymm12 d=ymm5 steps=ymm6,ymm7
# scale=ymm6,ymm7 pair_scale=ymm5 packed=ymm5,ymm6,ymm8,ymm10,ymm9 low_codes=ymm6,ymm8,ymm10,ymm12,ymm9,ymm7 signed_levels=ymm5,ymm6,ymm8,ymm10,ymm9,ymm1 unsigned_levels=ymm6,ymm8,ymm10,ymm12,ymm9,ymm7,ymm0
# x=ymm7,ymm8,ymm9,ymm10,ymm11,ymm12,ymm13,ymm14,ymm5 lanes=ymm7,ymm5,ymm9,ymm6,ymm8,ymm10,ymm0 group_scale=ymm10,ymm7,ymm12,ymm9,ymm11,ymm13,ymm8,ymm6,ymm1 zero=ymm0 high=ymm0 sum=ymm0
# pair=ymm1 result=ymm0 nothing=ymm0
    push rbp
    mov rbp, rsp
    and rsp, -128
    sub rsp, 128
    test rsi, rsi
    jz 4f
    # constants
    vmovdqa ymm0, ymmword ptr [rip + {c0}]
    vpbroadcastd ymm1, dword ptr [rip + {c1}]
    vpbroadcastd ymm2, dword ptr [rip + {c2}]
    vxorps xmm3, xmm3, xmm3
    vxorps xmm4, xmm4, xmm4
    lea rax, [rsp]
    lea r9, [rsp + 64]
    # scales of superblock +0
    vpbroadcastw xmm5, word ptr [rdi]
    vcvtph2ps ymm5, xmm5
    vmovd xmm6, dword ptr [rdi + 2]
    vpshufb xmm6, xmm6, xmmword ptr [rip + {c3}]
    vpmovzxbd ymm6, xmm6
    vpsrlvd ymm6, ymm6, ymmword ptr [rip + {c4}]
    vpand ymm6, ymm6, ymmword ptr [rip + {c5}]
    vpaddd ymm6, ymm6, ymm6
    vpor ymm6, ymm6, ymmword ptr [rip + {c6}]
    vcvtdq2ps ymm6, ymm6
    vmulps ymm6, ymm5, ymm6
    vshufps ymm5, ymm6, ymm6, 0x50
    vmulps ymm5, ymm5, ymmword ptr [rcx]
    vmovaps ymmword ptr [rax], ymm5
    vshufps ymm5, ymm6, ymm6, 0xfa
    vmulps ymm5, ymm5, ymmword ptr [rcx + 32]
    vmovaps ymmword ptr [rax + 32], ymm5
    # products of group 0 of superblock +0
    vmovdqu xmm5, xmmword ptr [rdi + 6]
    vpand xmm6, xmm5, xmm1
    vpsrlw xmm5, xmm5, 4
    vpand xmm5, xmm5, xmm1
    vinserti128 ymm5, ymm6, xmm5, 1
    vpshufb ymm5, ymm0, ymm5
    vpabsb ymm6, ymm5
    vmovdqa ymm7, ymmword ptr [rdx]
    vpsignb ymm7, ymm7, ymm5
    vpmaddubsw ymm7, ymm6, ymm7
    vmovdqa ymm8, ymmword ptr [rdx + 32]
    vpsignb ymm8, ymm8, ymm5
    vpmaddubsw ymm5, ymm6, ymm8
    # products of group 1 of superblock +0
    vmovdqu xmm6, xmmword ptr [rdi + 22]
    vpand xmm8, xmm6, xmm1
    vpsrlw xmm6, xmm6, 4
    vpand xmm6, xmm6, xmm1
    vinserti128 ymm6, ymm8, xmm6, 1
    vpshufb ymm6, ymm0, ymm6
    vpabsb ymm8, ymm6
    vmovdqa ymm9, ymmword ptr [rdx + 64]
    vpsignb ymm9, ymm9, ymm6
    vpmaddubsw ymm9, ymm8, ymm9
    vmovdqa ymm10, ymmword ptr [rdx + 96]
    vpsignb ymm10, ymm10, ymm6
    vpmaddubsw ymm6, ymm8, ymm10
    # products of group 2 of superblock +0
    vmovdqu xmm8, xmmword ptr [rdi + 38]
    vpand xmm10, xmm8, xmm1
    vpsrlw xmm8, xmm8, 4
    vpand xmm8, xmm8, xmm1
    vinserti128 ymm8, ymm10, xmm8, 1
    vpshufb ymm8, ymm0, ymm8
    vpabsb ymm10, ymm8
    vmovdqa ymm11, ymmword ptr [rdx + 128]
    vpsignb ymm11, ymm11, ymm8
    vpmaddubsw ymm11, ymm10, ymm11
    vmovdqa ymm12, ymmword ptr [rdx + 160]
    vpsignb ymm12, ymm12, ymm8
    vpmaddubsw ymm8, ymm10, ymm12
    dec rsi
    jz 3f
2:
    # sums of group 0
    vpmaddwd ymm7, ymm7, ymm2
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm10, dword ptr [rax]
    vfmadd231ps ymm3, ymm7, ymm10
    vpmaddwd ymm5, ymm5, ymm2
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm7, dword ptr [rax + 4]
    vfmadd231ps ymm4, ymm5, ymm7
    # scales of superblock +1
    vpbroadcastw xmm5, word ptr [rdi + 134]
    vcvtph2ps ymm5, xmm5
    vmovd xmm7, dword ptr [rdi + 136]
    # products of group 3 of superblock +0
    vmovdqu xmm10, xmmword ptr [rdi + 54]
    vpand xmm12, xmm10, xmm1
    vpsrlw xmm10, xmm10, 4
    vpand xmm10, xmm10, xmm1
    vinserti128 ymm10, ymm12, xmm10, 1
    vpshufb ymm10, ymm0, ymm10
    vpabsb ymm12, ymm10
    vmovdqa ymm13, ymmword ptr [rdx + 192]
    vpsignb ymm13, ymm13, ymm10
    vpmaddubsw ymm13, ymm12, ymm13
    vmovdqa ymm14, ymmword ptr [rdx + 224]
    vpsignb ymm14, ymm14, ymm10
    vpmaddubsw ymm10, ymm12, ymm14
    # sums of group 1
    vpmaddwd ymm9, ymm9, ymm2
    vcvtdq2ps ymm9, ymm9
    vbroadcastss ymm12, dword ptr [rax + 8]
    vfmadd231ps ymm3, ymm9, ymm12
    vpmaddwd ymm6, ymm6, ymm2
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm9, dword ptr [rax + 12]
    vfmadd231ps ymm4, ymm6, ymm9
    vpshufb xmm7, xmm7, xmmword ptr [rip + {c3}]
    vpmovzxbd ymm7, xmm7
    vpsrlvd ymm7, ymm7, ymmword ptr [rip + {c4}]
    # products of group 4 of superblock +0
    vmovdqu xmm6, xmmword ptr [rdi + 70]
    vpand xmm9, xmm6, xmm1
    vpsrlw xmm6, xmm6, 4
    vpand xmm6, xmm6, xmm1
    vinserti128 ymm6, ymm9, xmm6, 1
    vpshufb ymm6, ymm0, ymm6
    vpabsb ymm9, ymm6
    vmovdqa ymm12, ymmword ptr [rdx + 256]
    vpsignb ymm12, ymm12, ymm6
    vpmaddubsw ymm12, ymm9, ymm12
    vmovdqa ymm14, ymmword ptr [rdx + 288]
    vpsignb ymm14, ymm14, ymm6
    vpmaddubsw ymm6, ymm9, ymm14
    # sums of group 2
    vpmaddwd ymm9, ymm11, ymm2
    vcvtdq2ps ymm9, ymm9
    vbroadcastss ymm11, dword ptr [rax + 16]
    vfmadd231ps ymm3, ymm9, ymm11
    vpmaddwd ymm8, ymm8, ymm2
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm9, dword ptr [rax + 20]
    vfmadd231ps ymm4, ymm8, ymm9
    vpand ymm7, ymm7, ymmword ptr [rip + {c5}]
    vpaddd ymm7, ymm7, ymm7
    vpor ymm7, ymm7, ymmword ptr [rip + {c6}]
    # products of group 5 of superblock +0
    vmovdqu xmm8, xmmword ptr [rdi + 86]
    vpand xmm9, xmm8, xmm1
    vpsrlw xmm8, xmm8, 4
    vpand xmm8, xmm8, xmm1
    vinserti128 ymm8, ymm9, xmm8, 1
    vpshufb ymm8, ymm0, ymm8
    vpabsb ymm9, ymm8
    vmovdqa ymm11, ymmword ptr [rdx + 320]
    vpsignb ymm11, ymm11, ymm8
    vpmaddubsw ymm11, ymm9, ymm11
    vmovdqa ymm14, ymmword ptr [rdx + 352]
    vpsignb ymm14, ymm14, ymm8
    vpmaddubsw ymm8, ymm9, ymm14
    # sums of group 3
    vpmaddwd ymm9, ymm13, ymm2
    vcvtdq2ps ymm9, ymm9
    vbroadcastss ymm13, dword ptr [rax + 24]
    vfmadd231ps ymm3, ymm9, ymm13
    vpmaddwd ymm9, ymm10, ymm2
    vcvtdq2ps ymm9, ymm9
    vbroadcastss ymm10, dword ptr [rax + 28]
    vfmadd231ps ymm4, ymm9, ymm10
    vcvtdq2ps ymm7, ymm7
    vmulps ymm7, ymm5, ymm7
    vshufps ymm5, ymm7, ymm7, 0x50
    # products of group 6 of superblock +0
    vmovdqu xmm9, xmmword ptr [rdi + 102]
    vpand xmm10, xmm9, xmm1
    vpsrlw xmm9, xmm9, 4
    vpand xmm9, xmm9, xmm1
    vinserti128 ymm9, ymm10, xmm9, 1
    vpshufb ymm9, ymm0, ymm9
    vpabsb ymm10, ymm9
    vmovdqa ymm13, ymmword ptr [rdx + 384]
    vpsignb ymm13, ymm13, ymm9
    vpmaddubsw ymm13, ymm10, ymm13
    vmovdqa ymm14, ymmword ptr [rdx + 416]
    vpsignb ymm14, ymm14, ymm9
    vpmaddubsw ymm9, ymm10, ymm14
    # sums of group 4
    vpmaddwd ymm10, ymm12, ymm2
    vcvtdq2ps ymm10, ymm10
    vbroadcastss ymm12, dword ptr [rax + 32]
    vfmadd231ps ymm3, ymm10, ymm12
    vpmaddwd ymm6, ymm6, ymm2
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm10, dword ptr [rax + 36]
    vfmadd231ps ymm4, ymm6, ymm10
    vmulps ymm5, ymm5, ymmword ptr [rcx + 64]
    vmovaps ymmword ptr [r9], ymm5
    vshufps ymm5, ymm7, ymm7, 0xfa
    # products of group 7 of superblock +0
    vmovdqu xmm6, xmmword ptr [rdi + 118]
    vpand xmm7, xmm6, xmm1
    vpsrlw xmm6, xmm6, 4
    vpand xmm6, xmm6, xmm1
    vinserti128 ymm6, ymm7, xmm6, 1
    vpshufb ymm6, ymm0, ymm6
    vpabsb ymm7, ymm6
    vmovdqa ymm10, ymmword ptr [rdx + 448]
    vpsignb ymm10, ymm10, ymm6
    vpmaddubsw ymm10, ymm7, ymm10
    vmovdqa ymm12, ymmword ptr [rdx + 480]
    vpsignb ymm12, ymm12, ymm6
    vpmaddubsw ymm12, ymm7, ymm12
    # sums of group 5
    vpmaddwd ymm6, ymm11, ymm2
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 40]
    vfmadd231ps ymm3, ymm6, ymm7
    vpmaddwd ymm6, ymm8, ymm2
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 44]
    vfmadd231ps ymm4, ymm6, ymm7
    vmulps ymm5, ymm5, ymmword ptr [rcx + 96]
    vmovaps ymmword ptr [r9 + 32], ymm5
    # products of group 0 of superblock +1
    vmovdqu xmm5, xmmword ptr [rdi + 140]
    vpand xmm6, xmm5, xmm1
    vpsrlw xmm5, xmm5, 4
    vpand xmm5, xmm5, xmm1
    vinserti128 ymm5, ymm6, xmm5, 1
    vpshufb ymm5, ymm0, ymm5
    vpabsb ymm6, ymm5
    vmovdqa ymm7, ymmword ptr [rdx + 512]
    vpsignb ymm7, ymm7, ymm5
    vpmaddubsw ymm7, ymm6, ymm7
    vmovdqa ymm8, ymmword ptr [rdx + 544]
    vpsignb ymm8, ymm8, ymm5
    vpmaddubsw ymm5, ymm6, ymm8
    # sums of group 6
    vpmaddwd ymm6, ymm13, ymm2
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm8, dword ptr [rax + 48]
    vfmadd231ps ymm3, ymm6, ymm8
    vpmaddwd ymm6, ymm9, ymm2
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm8, dword ptr [rax + 52]
    vfmadd231ps ymm4, ymm6, ymm8
    # products of group 1 of superblock +1
    vmovdqu xmm6, xmmword ptr [rdi + 156]
    vpand xmm8, xmm6, xmm1
    vpsrlw xmm6, xmm6, 4
    vpand xmm6, xmm6, xmm1
    vinserti128 ymm6, ymm8, xmm6, 1
    vpshufb ymm6, ymm0, ymm6
    vpabsb ymm8, ymm6
    vmovdqa ymm9, ymmword ptr [rdx + 576]
    vpsignb ymm9, ymm9, ymm6
    vpmaddubsw ymm9, ymm8, ymm9
    vmovdqa ymm11, ymmword ptr [rdx + 608]
    vpsignb ymm11, ymm11, ymm6
    vpmaddubsw ymm6, ymm8, ymm11
    # sums of group 7
    vpmaddwd ymm8, ymm10, ymm2
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm10, dword ptr [rax + 56]
    vfmadd231ps ymm3, ymm8, ymm10
    vpmaddwd ymm8, ymm12, ymm2
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm10, dword ptr [rax + 60]
    vfmadd231ps ymm4, ymm8, ymm10
    # products of group 2 of superblock +1
    vmovdqu xmm8, xmmword ptr [rdi + 172]
    vpand xmm10, xmm8, xmm1
    vpsrlw xmm8, xmm8, 4
    vpand xmm8, xmm8, xmm1
    vinserti128 ymm8, ymm10, xmm8, 1
    vpshufb ymm8, ymm0, ymm8
    vpabsb ymm10, ymm8
    vmovdqa ymm11, ymmword ptr [rdx + 640]
    vpsignb ymm11, ymm11, ymm8
    vpmaddubsw ymm11, ymm10, ymm11
    vmovdqa ymm12, ymmword ptr [rdx + 672]
    vpsignb ymm12, ymm12, ymm8
    vpmaddubsw ymm8, ymm10, ymm12
    add rdi, 134
    add rdx, 512
    add rcx, 64
    xor rax, 64
    xor r9, 64
    dec rsi
    jnz 2b
3:
    # sums of group 0
    vpmaddwd ymm7, ymm7, ymm2
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm10, dword ptr [rax]
    vfmadd231ps ymm3, ymm7, ymm10
    vpmaddwd ymm5, ymm5, ymm2
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm7, dword ptr [rax + 4]
    vfmadd231ps ymm4, ymm5, ymm7
    # products of group 3 of superblock +0
    vmovdqu xmm5, xmmword ptr [rdi + 54]
    vpand xmm7, xmm5, xmm1
    vpsrlw xmm5, xmm5, 4
    vpand xmm5, xmm5, xmm1
    vinserti128 ymm5, ymm7, xmm5, 1
    vpshufb ymm5, ymm0, ymm5
    vpabsb ymm7, ymm5
    vmovdqa ymm10, ymmword ptr [rdx + 192]
    vpsignb ymm10, ymm10, ymm5
    vpmaddubsw ymm13, ymm7, ymm10
    vmovdqa ymm10, ymmword ptr [rdx + 224]
    vpsignb ymm10, ymm10, ymm5
    vpmaddubsw ymm10, ymm7, ymm10
    # sums of group 1
    vpmaddwd ymm5, ymm9, ymm2
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm7, dword ptr [rax + 8]
    vfmadd231ps ymm3, ymm5, ymm7
    vpmaddwd ymm5, ymm6, ymm2
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 12]
    vfmadd231ps ymm4, ymm5, ymm6
    # products of group 4 of superblock +0
    vmovdqu xmm5, xmmword ptr [rdi + 70]
    vpand xmm6, xmm5, xmm1
    vpsrlw xmm5, xmm5, 4
    vpand xmm5, xmm5, xmm1
    vinserti128 ymm5, ymm6, xmm5, 1
    vpshufb ymm5, ymm0, ymm5
    vpabsb ymm6, ymm5
    vmovdqa ymm7, ymmword ptr [rdx + 256]
    vpsignb ymm7, ymm7, ymm5
    vpmaddubsw ymm12, ymm6, ymm7
    vmovdqa ymm7, ymmword ptr [rdx + 288]
    vpsignb ymm7, ymm7, ymm5
    vpmaddubsw ymm6, ymm6, ymm7
    # sums of group 2
    vpmaddwd ymm5, ymm11, ymm2
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm7, dword ptr [rax + 16]
    vfmadd231ps ymm3, ymm5, ymm7
    vpmaddwd ymm5, ymm8, ymm2
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm7, dword ptr [rax + 20]
    vfmadd231ps ymm4, ymm5, ymm7
    # products of group 5 of superblock +0
    vmovdqu xmm5, xmmword ptr [rdi + 86]
    vpand xmm7, xmm5, xmm1
    vpsrlw xmm5, xmm5, 4
    vpand xmm5, xmm5, xmm1
    vinserti128 ymm5, ymm7, xmm5, 1
    vpshufb ymm5, ymm0, ymm5
    vpabsb ymm7, ymm5
    vmovdqa ymm8, ymmword ptr [rdx + 320]
    vpsignb ymm8, ymm8, ymm5
    vpmaddubsw ymm11, ymm7, ymm8
    vmovdqa ymm8, ymmword ptr [rdx + 352]
    vpsignb ymm8, ymm8, ymm5
    vpmaddubsw ymm8, ymm7, ymm8
    # sums of group 3
    vpmaddwd ymm5, ymm13, ymm2
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm7, dword ptr [rax + 24]
    vfmadd231ps ymm3, ymm5, ymm7
    vpmaddwd ymm5, ymm10, ymm2
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm7, dword ptr [rax + 28]
    vfmadd231ps ymm4, ymm5, ymm7
    # products of group 6 of superblock +0
    vmovdqu xmm5, xmmword ptr [rdi + 102]
    vpand xmm7, xmm5, xmm1
    vpsrlw xmm5, xmm5, 4
    vpand xmm5, xmm5, xmm1
    vinserti128 ymm5, ymm7, xmm5, 1
    vpshufb ymm5, ymm0, ymm5
    vpabsb ymm7, ymm5
    vmovdqa ymm9, ymmword ptr [rdx + 384]
    vpsignb ymm9, ymm9, ymm5
    vpmaddubsw ymm13, ymm7, ymm9
    vmovdqa ymm9, ymmword ptr [rdx + 416]
    vpsignb ymm9, ymm9, ymm5
    vpmaddubsw ymm9, ymm7, ymm9
    # sums of group 4
    vpmaddwd ymm5, ymm12, ymm2
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm7, dword ptr [rax + 32]
    vfmadd231ps ymm3, ymm5, ymm7
    vpmaddwd ymm5, ymm6, ymm2
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 36]
    vfmadd231ps ymm4, ymm5, ymm6
    # products of group 7 of superblock +0
    vmovdqu xmm5, xmmword ptr [rdi + 118]
    vpand xmm6, xmm5, xmm1
    vpsrlw xmm5, xmm5, 4
    vpand xmm5, xmm5, xmm1
    vinserti128 ymm1, ymm6, xmm5, 1
    vpshufb ymm1, ymm0, ymm1
    vpabsb ymm0, ymm1
    vmovdqa ymm5, ymmword ptr [rdx + 448]
    vpsignb ymm5, ymm5, ymm1
    vpmaddubsw ymm10, ymm0, ymm5
    vmovdqa ymm5, ymmword ptr [rdx + 480]
    vpsignb ymm5, ymm5, ymm1
    vpmaddubsw ymm12, ymm0, ymm5
    # sums of group 5
    vpmaddwd ymm0, ymm11, ymm2
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm1, dword ptr [rax + 40]
    vfmadd231ps ymm3, ymm0, ymm1
    vpmaddwd ymm0, ymm8, ymm2
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm1, dword ptr [rax + 44]
    vfmadd231ps ymm4, ymm0, ymm1
    # sums of group 6
    vpmaddwd ymm0, ymm13, ymm2
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm1, dword ptr [rax + 48]
    vfmadd231ps ymm3, ymm0, ymm1
    vpmaddwd ymm0, ymm9, ymm2
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm1, dword ptr [rax + 52]
    vfmadd231ps ymm4, ymm0, ymm1
    # sums of group 7
    vpmaddwd ymm0, ymm10, ymm2
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm1, dword ptr [rax + 56]
    vfmadd231ps ymm3, ymm0, ymm1
    vpmaddwd ymm0, ymm12, ymm2
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm1, dword ptr [rax + 60]
    vfmadd231ps ymm4, ymm0, ymm1
    # the lanes summed
    vxorps xmm0, xmm0, xmm0
    vaddps ymm3, ymm3, ymm0
    vextractf128 xmm0, ymm3, 1
    vaddps xmm0, xmm3, xmm0
    vmovhlps xmm1, xmm0, xmm0
    vaddps xmm0, xmm0, xmm1
    vmovshdup xmm1, xmm0
    vaddss xmm0, xmm0, xmm1
    vmovss dword ptr [r8], xmm0
    # the lanes summed
    vxorps xmm0, xmm0, xmm0
    vaddps ymm4, ymm4, ymm0
    vextractf128 xmm0, ymm4, 1
    vaddps xmm0, xmm4, xmm0
    vmovhlps xmm1, xmm0, xmm0
    vaddps xmm0, xmm0, xmm1
    vmovshdup xmm1, xmm0
    vaddss xmm0, xmm0, xmm1
    vmovss dword ptr [r8 + 4], xmm0
    mov rsp, rbp
    pop rbp
    vzeroupper
    ret
4:
    vxorps xmm0, xmm0, xmm0
    vmovss dword ptr [r8], xmm0
    vmovss dword ptr [r8 + 4], xmm0
    mov rsp, rbp
    pop rbp
    vzeroupper
    ret
# c0, aligned to 32:
#   01 03 05 07 09 0b 0d 0f ff fd fb f9 f7 f5 f3 f1
#   01 03 05 07 09 0b 0d 0f ff fd fb f9 f7 f5 f3 f1
# c1, aligned to 4:
#   0f 0f 0f 0f
# c2, aligned to 4:
#   01 00 01 00
# c3, aligned to 16:
#   00 00 02 02 01 01 03 03 80 80 80 80 80 80 80 80
# c4, aligned to 32:
#   00 00 00 00 04 00 00 00 00 00 00 00 04 00 00 00
#   00 00 00 00 04 00 00 00 00 00 00 00 04 00 00 00
# c5, aligned to 32:
#   0f 00 00 00 0f 00 00 00 0f 00 00 00 0f 00 00 00
#   0f 00 00 00 0f 00 00 00 0f 00 00 00 0f 00 00 00
# c6, aligned to 32:
#   01 00 00 00 01 00 00 00 01 00 00 00 01 00 00 00
#   01 00 00 00 01 00 00 00 01 00 00 00 01 00 00 00
