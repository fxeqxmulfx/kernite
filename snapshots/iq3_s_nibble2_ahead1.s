# iq3_s_nibble2_ahead1(gpr, gpr, gpr, gpr, gpr) -> ()
# frame: 128 bytes aligned to 128, saves []
# row=rdi sbs=rsi quants=rdx scales=rcx out=r8 levels=ymm0
# nibble=ymm1 one16=ymm2 acc0=ymm3 acc1=ymm4 cur=rax nxt=r9
# p0a=ymm7 p0b=ymm5 p1a=ymm9 p1b=ymm7 p2a=ymm9 p2b=ymm7
# p3a=ymm9 p3b=ymm7 p4a=ymm9 p4b=ymm7 p5a=ymm8 p5b=ymm6
# p6a=ymm7 p6b=ymm5 p7a=ymm7 p7b=ymm5 d=ymm5 steps=ymm6
# scale=ymm6 pair_scale=ymm5 packed=ymm5,ymm7,ymm6 low_codes=ymm6,ymm8,ymm7 signed_levels=ymm5,ymm7,ymm6,ymm1 unsigned_levels=ymm6,ymm8,ymm7,ymm0
# x=ymm7,ymm8,ymm9,ymm10,ymm5 lanes=ymm6,ymm5,ymm8,ymm7,ymm0 group_scale=ymm7,ymm6,ymm9,ymm8,ymm1 zero=ymm0 high=ymm0 sum=ymm0
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
    dec rsi
    jz 3f
2:
    # sums of group 0
    vpmaddwd ymm6, ymm7, ymm2
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax]
    vfmadd231ps ymm3, ymm6, ymm7
    vpmaddwd ymm5, ymm5, ymm2
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 4]
    vfmadd231ps ymm4, ymm5, ymm6
    # scales of superblock +1
    vpbroadcastw xmm5, word ptr [rdi + 134]
    vcvtph2ps ymm5, xmm5
    vmovd xmm6, dword ptr [rdi + 136]
    # products of group 1 of superblock +0
    vmovdqu xmm7, xmmword ptr [rdi + 22]
    vpand xmm8, xmm7, xmm1
    vpsrlw xmm7, xmm7, 4
    vpand xmm7, xmm7, xmm1
    vinserti128 ymm7, ymm8, xmm7, 1
    vpshufb ymm7, ymm0, ymm7
    vpabsb ymm8, ymm7
    vmovdqa ymm9, ymmword ptr [rdx + 64]
    vpsignb ymm9, ymm9, ymm7
    vpmaddubsw ymm9, ymm8, ymm9
    vmovdqa ymm10, ymmword ptr [rdx + 96]
    vpsignb ymm10, ymm10, ymm7
    vpmaddubsw ymm7, ymm8, ymm10
    # sums of group 1
    vpmaddwd ymm8, ymm9, ymm2
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm9, dword ptr [rax + 8]
    vfmadd231ps ymm3, ymm8, ymm9
    vpmaddwd ymm7, ymm7, ymm2
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rax + 12]
    vfmadd231ps ymm4, ymm7, ymm8
    vpshufb xmm6, xmm6, xmmword ptr [rip + {c3}]
    vpmovzxbd ymm6, xmm6
    vpsrlvd ymm6, ymm6, ymmword ptr [rip + {c4}]
    # products of group 2 of superblock +0
    vmovdqu xmm7, xmmword ptr [rdi + 38]
    vpand xmm8, xmm7, xmm1
    vpsrlw xmm7, xmm7, 4
    vpand xmm7, xmm7, xmm1
    vinserti128 ymm7, ymm8, xmm7, 1
    vpshufb ymm7, ymm0, ymm7
    vpabsb ymm8, ymm7
    vmovdqa ymm9, ymmword ptr [rdx + 128]
    vpsignb ymm9, ymm9, ymm7
    vpmaddubsw ymm9, ymm8, ymm9
    vmovdqa ymm10, ymmword ptr [rdx + 160]
    vpsignb ymm10, ymm10, ymm7
    vpmaddubsw ymm7, ymm8, ymm10
    # sums of group 2
    vpmaddwd ymm8, ymm9, ymm2
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm9, dword ptr [rax + 16]
    vfmadd231ps ymm3, ymm8, ymm9
    vpmaddwd ymm7, ymm7, ymm2
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rax + 20]
    vfmadd231ps ymm4, ymm7, ymm8
    vpand ymm6, ymm6, ymmword ptr [rip + {c5}]
    vpaddd ymm6, ymm6, ymm6
    vpor ymm6, ymm6, ymmword ptr [rip + {c6}]
    # products of group 3 of superblock +0
    vmovdqu xmm7, xmmword ptr [rdi + 54]
    vpand xmm8, xmm7, xmm1
    vpsrlw xmm7, xmm7, 4
    vpand xmm7, xmm7, xmm1
    vinserti128 ymm7, ymm8, xmm7, 1
    vpshufb ymm7, ymm0, ymm7
    vpabsb ymm8, ymm7
    vmovdqa ymm9, ymmword ptr [rdx + 192]
    vpsignb ymm9, ymm9, ymm7
    vpmaddubsw ymm9, ymm8, ymm9
    vmovdqa ymm10, ymmword ptr [rdx + 224]
    vpsignb ymm10, ymm10, ymm7
    vpmaddubsw ymm7, ymm8, ymm10
    # sums of group 3
    vpmaddwd ymm8, ymm9, ymm2
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm9, dword ptr [rax + 24]
    vfmadd231ps ymm3, ymm8, ymm9
    vpmaddwd ymm7, ymm7, ymm2
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rax + 28]
    vfmadd231ps ymm4, ymm7, ymm8
    vcvtdq2ps ymm6, ymm6
    vmulps ymm6, ymm5, ymm6
    vshufps ymm5, ymm6, ymm6, 0x50
    # products of group 4 of superblock +0
    vmovdqu xmm7, xmmword ptr [rdi + 70]
    vpand xmm8, xmm7, xmm1
    vpsrlw xmm7, xmm7, 4
    vpand xmm7, xmm7, xmm1
    vinserti128 ymm7, ymm8, xmm7, 1
    vpshufb ymm7, ymm0, ymm7
    vpabsb ymm8, ymm7
    vmovdqa ymm9, ymmword ptr [rdx + 256]
    vpsignb ymm9, ymm9, ymm7
    vpmaddubsw ymm9, ymm8, ymm9
    vmovdqa ymm10, ymmword ptr [rdx + 288]
    vpsignb ymm10, ymm10, ymm7
    vpmaddubsw ymm7, ymm8, ymm10
    # sums of group 4
    vpmaddwd ymm8, ymm9, ymm2
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm9, dword ptr [rax + 32]
    vfmadd231ps ymm3, ymm8, ymm9
    vpmaddwd ymm7, ymm7, ymm2
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rax + 36]
    vfmadd231ps ymm4, ymm7, ymm8
    vmulps ymm5, ymm5, ymmword ptr [rcx + 64]
    vmovaps ymmword ptr [r9], ymm5
    vshufps ymm5, ymm6, ymm6, 0xfa
    # products of group 5 of superblock +0
    vmovdqu xmm6, xmmword ptr [rdi + 86]
    vpand xmm7, xmm6, xmm1
    vpsrlw xmm6, xmm6, 4
    vpand xmm6, xmm6, xmm1
    vinserti128 ymm6, ymm7, xmm6, 1
    vpshufb ymm6, ymm0, ymm6
    vpabsb ymm7, ymm6
    vmovdqa ymm8, ymmword ptr [rdx + 320]
    vpsignb ymm8, ymm8, ymm6
    vpmaddubsw ymm8, ymm7, ymm8
    vmovdqa ymm9, ymmword ptr [rdx + 352]
    vpsignb ymm9, ymm9, ymm6
    vpmaddubsw ymm6, ymm7, ymm9
    # sums of group 5
    vpmaddwd ymm7, ymm8, ymm2
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rax + 40]
    vfmadd231ps ymm3, ymm7, ymm8
    vpmaddwd ymm6, ymm6, ymm2
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 44]
    vfmadd231ps ymm4, ymm6, ymm7
    vmulps ymm5, ymm5, ymmword ptr [rcx + 96]
    vmovaps ymmword ptr [r9 + 32], ymm5
    # products of group 6 of superblock +0
    vmovdqu xmm5, xmmword ptr [rdi + 102]
    vpand xmm6, xmm5, xmm1
    vpsrlw xmm5, xmm5, 4
    vpand xmm5, xmm5, xmm1
    vinserti128 ymm5, ymm6, xmm5, 1
    vpshufb ymm5, ymm0, ymm5
    vpabsb ymm6, ymm5
    vmovdqa ymm7, ymmword ptr [rdx + 384]
    vpsignb ymm7, ymm7, ymm5
    vpmaddubsw ymm7, ymm6, ymm7
    vmovdqa ymm8, ymmword ptr [rdx + 416]
    vpsignb ymm8, ymm8, ymm5
    vpmaddubsw ymm5, ymm6, ymm8
    # sums of group 6
    vpmaddwd ymm6, ymm7, ymm2
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 48]
    vfmadd231ps ymm3, ymm6, ymm7
    vpmaddwd ymm5, ymm5, ymm2
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 52]
    vfmadd231ps ymm4, ymm5, ymm6
    # products of group 7 of superblock +0
    vmovdqu xmm5, xmmword ptr [rdi + 118]
    vpand xmm6, xmm5, xmm1
    vpsrlw xmm5, xmm5, 4
    vpand xmm5, xmm5, xmm1
    vinserti128 ymm5, ymm6, xmm5, 1
    vpshufb ymm5, ymm0, ymm5
    vpabsb ymm6, ymm5
    vmovdqa ymm7, ymmword ptr [rdx + 448]
    vpsignb ymm7, ymm7, ymm5
    vpmaddubsw ymm7, ymm6, ymm7
    vmovdqa ymm8, ymmword ptr [rdx + 480]
    vpsignb ymm8, ymm8, ymm5
    vpmaddubsw ymm5, ymm6, ymm8
    # sums of group 7
    vpmaddwd ymm6, ymm7, ymm2
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 56]
    vfmadd231ps ymm3, ymm6, ymm7
    vpmaddwd ymm5, ymm5, ymm2
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 60]
    vfmadd231ps ymm4, ymm5, ymm6
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
    add rdi, 134
    add rdx, 512
    add rcx, 64
    xor rax, 64
    xor r9, 64
    dec rsi
    jnz 2b
3:
    # sums of group 0
    vpmaddwd ymm6, ymm7, ymm2
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax]
    vfmadd231ps ymm3, ymm6, ymm7
    vpmaddwd ymm5, ymm5, ymm2
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 4]
    vfmadd231ps ymm4, ymm5, ymm6
    # products of group 1 of superblock +0
    vmovdqu xmm5, xmmword ptr [rdi + 22]
    vpand xmm6, xmm5, xmm1
    vpsrlw xmm5, xmm5, 4
    vpand xmm5, xmm5, xmm1
    vinserti128 ymm5, ymm6, xmm5, 1
    vpshufb ymm5, ymm0, ymm5
    vpabsb ymm6, ymm5
    vmovdqa ymm7, ymmword ptr [rdx + 64]
    vpsignb ymm7, ymm7, ymm5
    vpmaddubsw ymm9, ymm6, ymm7
    vmovdqa ymm7, ymmword ptr [rdx + 96]
    vpsignb ymm7, ymm7, ymm5
    vpmaddubsw ymm7, ymm6, ymm7
    # sums of group 1
    vpmaddwd ymm5, ymm9, ymm2
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 8]
    vfmadd231ps ymm3, ymm5, ymm6
    vpmaddwd ymm5, ymm7, ymm2
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 12]
    vfmadd231ps ymm4, ymm5, ymm6
    # products of group 2 of superblock +0
    vmovdqu xmm5, xmmword ptr [rdi + 38]
    vpand xmm6, xmm5, xmm1
    vpsrlw xmm5, xmm5, 4
    vpand xmm5, xmm5, xmm1
    vinserti128 ymm5, ymm6, xmm5, 1
    vpshufb ymm5, ymm0, ymm5
    vpabsb ymm6, ymm5
    vmovdqa ymm7, ymmword ptr [rdx + 128]
    vpsignb ymm7, ymm7, ymm5
    vpmaddubsw ymm9, ymm6, ymm7
    vmovdqa ymm7, ymmword ptr [rdx + 160]
    vpsignb ymm7, ymm7, ymm5
    vpmaddubsw ymm7, ymm6, ymm7
    # sums of group 2
    vpmaddwd ymm5, ymm9, ymm2
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 16]
    vfmadd231ps ymm3, ymm5, ymm6
    vpmaddwd ymm5, ymm7, ymm2
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 20]
    vfmadd231ps ymm4, ymm5, ymm6
    # products of group 3 of superblock +0
    vmovdqu xmm5, xmmword ptr [rdi + 54]
    vpand xmm6, xmm5, xmm1
    vpsrlw xmm5, xmm5, 4
    vpand xmm5, xmm5, xmm1
    vinserti128 ymm5, ymm6, xmm5, 1
    vpshufb ymm5, ymm0, ymm5
    vpabsb ymm6, ymm5
    vmovdqa ymm7, ymmword ptr [rdx + 192]
    vpsignb ymm7, ymm7, ymm5
    vpmaddubsw ymm9, ymm6, ymm7
    vmovdqa ymm7, ymmword ptr [rdx + 224]
    vpsignb ymm7, ymm7, ymm5
    vpmaddubsw ymm7, ymm6, ymm7
    # sums of group 3
    vpmaddwd ymm5, ymm9, ymm2
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 24]
    vfmadd231ps ymm3, ymm5, ymm6
    vpmaddwd ymm5, ymm7, ymm2
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 28]
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
    vpmaddubsw ymm9, ymm6, ymm7
    vmovdqa ymm7, ymmword ptr [rdx + 288]
    vpsignb ymm7, ymm7, ymm5
    vpmaddubsw ymm7, ymm6, ymm7
    # sums of group 4
    vpmaddwd ymm5, ymm9, ymm2
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 32]
    vfmadd231ps ymm3, ymm5, ymm6
    vpmaddwd ymm5, ymm7, ymm2
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 36]
    vfmadd231ps ymm4, ymm5, ymm6
    # products of group 5 of superblock +0
    vmovdqu xmm5, xmmword ptr [rdi + 86]
    vpand xmm6, xmm5, xmm1
    vpsrlw xmm5, xmm5, 4
    vpand xmm5, xmm5, xmm1
    vinserti128 ymm5, ymm6, xmm5, 1
    vpshufb ymm5, ymm0, ymm5
    vpabsb ymm6, ymm5
    vmovdqa ymm7, ymmword ptr [rdx + 320]
    vpsignb ymm7, ymm7, ymm5
    vpmaddubsw ymm8, ymm6, ymm7
    vmovdqa ymm7, ymmword ptr [rdx + 352]
    vpsignb ymm7, ymm7, ymm5
    vpmaddubsw ymm6, ymm6, ymm7
    # sums of group 5
    vpmaddwd ymm5, ymm8, ymm2
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm7, dword ptr [rax + 40]
    vfmadd231ps ymm3, ymm5, ymm7
    vpmaddwd ymm5, ymm6, ymm2
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 44]
    vfmadd231ps ymm4, ymm5, ymm6
    # products of group 6 of superblock +0
    vmovdqu xmm5, xmmword ptr [rdi + 102]
    vpand xmm6, xmm5, xmm1
    vpsrlw xmm5, xmm5, 4
    vpand xmm5, xmm5, xmm1
    vinserti128 ymm5, ymm6, xmm5, 1
    vpshufb ymm5, ymm0, ymm5
    vpabsb ymm6, ymm5
    vmovdqa ymm7, ymmword ptr [rdx + 384]
    vpsignb ymm7, ymm7, ymm5
    vpmaddubsw ymm7, ymm6, ymm7
    vmovdqa ymm8, ymmword ptr [rdx + 416]
    vpsignb ymm8, ymm8, ymm5
    vpmaddubsw ymm5, ymm6, ymm8
    # sums of group 6
    vpmaddwd ymm6, ymm7, ymm2
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 48]
    vfmadd231ps ymm3, ymm6, ymm7
    vpmaddwd ymm5, ymm5, ymm2
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 52]
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
    vpmaddubsw ymm7, ymm0, ymm5
    vmovdqa ymm5, ymmword ptr [rdx + 480]
    vpsignb ymm5, ymm5, ymm1
    vpmaddubsw ymm5, ymm0, ymm5
    # sums of group 7
    vpmaddwd ymm0, ymm7, ymm2
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm1, dword ptr [rax + 56]
    vfmadd231ps ymm3, ymm0, ymm1
    vpmaddwd ymm0, ymm5, ymm2
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
