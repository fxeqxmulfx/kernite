# q6_k(gpr, gpr, gpr, gpr, gpr) -> vec
# frame: 0 bytes aligned to 0, saves []
# row=rdi superblocks=rsi quants=rdx scales=rcx half_sums=r8 nibble_mask=ymm0
# pair_mask=ymm1 one16=ymm2 acc=ymm3 off=ymm4 half_d=ymm5 d=ymm5
# ql0=ymm6 ql1=ymm7 qh=ymm8 nib=ymm9,ymm6 high=ymm10,ymm9,ymm7,ymm0 x=ymm10,ymm9,ymm7
# pairs=ymm9,ymm6 pair_scales=ymm10,ymm9,ymm7 ws=ymm10,ymm9,ymm7 dx=ymm11,ymm10,ymm8 eight_scales=ymm6 wide_scales=ymm6
# dsums=ymm7,ymm5 sum=ymm0 pair=ymm1 result=ymm0 zero=ymm0
    test rsi, rsi
    jz 3f
    vpbroadcastd ymm0, dword ptr [rip + {c0}]
    vpbroadcastd ymm1, dword ptr [rip + {c1}]
    vpbroadcastd ymm2, dword ptr [rip + {c2}]
    vxorps xmm3, xmm3, xmm3
    vxorps xmm4, xmm4, xmm4
2:
    vpbroadcastw xmm5, word ptr [rdi + 208]
    vcvtph2ps ymm5, xmm5
    vmovdqu ymm6, ymmword ptr [rdi]
    vmovdqu ymm7, ymmword ptr [rdi + 32]
    vmovdqu ymm8, ymmword ptr [rdi + 128]
    vpand ymm9, ymm6, ymm0
    vpand ymm10, ymm8, ymm1
    vpsllw ymm10, ymm10, 4
    vpor ymm9, ymm9, ymm10
    vmovdqa ymm10, ymmword ptr [rdx]
    vpmaddubsw ymm9, ymm9, ymm10
    vpmaddwd ymm9, ymm9, ymm2
    vcvtdq2ps ymm9, ymm9
    vmovd xmm10, dword ptr [rdi + 192]
    vpshufb xmm10, xmm10, xmmword ptr [rip + {c3}]
    vpmovsxbd ymm10, xmm10
    vcvtdq2ps ymm10, ymm10
    vmulps ymm10, ymm10, ymm5
    vbroadcastss ymm11, dword ptr [rcx]
    vmulps ymm10, ymm10, ymm11
    vfmadd231ps ymm3, ymm9, ymm10
    vpand ymm9, ymm7, ymm0
    vpsrlw ymm10, ymm8, 2
    vpand ymm10, ymm10, ymm1
    vpsllw ymm10, ymm10, 4
    vpor ymm9, ymm9, ymm10
    vmovdqa ymm10, ymmword ptr [rdx + 32]
    vpmaddubsw ymm9, ymm9, ymm10
    vpmaddwd ymm9, ymm9, ymm2
    vcvtdq2ps ymm9, ymm9
    vmovd xmm10, dword ptr [rdi + 194]
    vpshufb xmm10, xmm10, xmmword ptr [rip + {c3}]
    vpmovsxbd ymm10, xmm10
    vcvtdq2ps ymm10, ymm10
    vmulps ymm10, ymm10, ymm5
    vbroadcastss ymm11, dword ptr [rcx + 4]
    vmulps ymm10, ymm10, ymm11
    vfmadd231ps ymm3, ymm9, ymm10
    vpsrlw ymm6, ymm6, 4
    vpand ymm6, ymm6, ymm0
    vpsrlw ymm9, ymm8, 4
    vpand ymm9, ymm9, ymm1
    vpsllw ymm9, ymm9, 4
    vpor ymm6, ymm6, ymm9
    vmovdqa ymm9, ymmword ptr [rdx + 64]
    vpmaddubsw ymm6, ymm6, ymm9
    vpmaddwd ymm6, ymm6, ymm2
    vcvtdq2ps ymm6, ymm6
    vmovd xmm9, dword ptr [rdi + 196]
    vpshufb xmm9, xmm9, xmmword ptr [rip + {c3}]
    vpmovsxbd ymm9, xmm9
    vcvtdq2ps ymm9, ymm9
    vmulps ymm9, ymm9, ymm5
    vbroadcastss ymm10, dword ptr [rcx + 8]
    vmulps ymm9, ymm9, ymm10
    vfmadd231ps ymm3, ymm6, ymm9
    vpsrlw ymm6, ymm7, 4
    vpand ymm6, ymm6, ymm0
    vpsrlw ymm7, ymm8, 6
    vpand ymm7, ymm7, ymm1
    vpsllw ymm7, ymm7, 4
    vpor ymm6, ymm6, ymm7
    vmovdqa ymm7, ymmword ptr [rdx + 96]
    vpmaddubsw ymm6, ymm6, ymm7
    vpmaddwd ymm6, ymm6, ymm2
    vcvtdq2ps ymm6, ymm6
    vmovd xmm7, dword ptr [rdi + 198]
    vpshufb xmm7, xmm7, xmmword ptr [rip + {c3}]
    vpmovsxbd ymm7, xmm7
    vcvtdq2ps ymm7, ymm7
    vmulps ymm7, ymm7, ymm5
    vbroadcastss ymm8, dword ptr [rcx + 12]
    vmulps ymm7, ymm7, ymm8
    vfmadd231ps ymm3, ymm6, ymm7
    vmovdqu ymm6, ymmword ptr [rdi + 64]
    vmovdqu ymm7, ymmword ptr [rdi + 96]
    vmovdqu ymm8, ymmword ptr [rdi + 160]
    vpand ymm9, ymm6, ymm0
    vpand ymm10, ymm8, ymm1
    vpsllw ymm10, ymm10, 4
    vpor ymm9, ymm9, ymm10
    vmovdqa ymm10, ymmword ptr [rdx + 128]
    vpmaddubsw ymm9, ymm9, ymm10
    vpmaddwd ymm9, ymm9, ymm2
    vcvtdq2ps ymm9, ymm9
    vmovd xmm10, dword ptr [rdi + 200]
    vpshufb xmm10, xmm10, xmmword ptr [rip + {c3}]
    vpmovsxbd ymm10, xmm10
    vcvtdq2ps ymm10, ymm10
    vmulps ymm10, ymm10, ymm5
    vbroadcastss ymm11, dword ptr [rcx + 16]
    vmulps ymm10, ymm10, ymm11
    vfmadd231ps ymm3, ymm9, ymm10
    vpand ymm9, ymm7, ymm0
    vpsrlw ymm10, ymm8, 2
    vpand ymm10, ymm10, ymm1
    vpsllw ymm10, ymm10, 4
    vpor ymm9, ymm9, ymm10
    vmovdqa ymm10, ymmword ptr [rdx + 160]
    vpmaddubsw ymm9, ymm9, ymm10
    vpmaddwd ymm9, ymm9, ymm2
    vcvtdq2ps ymm9, ymm9
    vmovd xmm10, dword ptr [rdi + 202]
    vpshufb xmm10, xmm10, xmmword ptr [rip + {c3}]
    vpmovsxbd ymm10, xmm10
    vcvtdq2ps ymm10, ymm10
    vmulps ymm10, ymm10, ymm5
    vbroadcastss ymm11, dword ptr [rcx + 20]
    vmulps ymm10, ymm10, ymm11
    vfmadd231ps ymm3, ymm9, ymm10
    vpsrlw ymm6, ymm6, 4
    vpand ymm6, ymm6, ymm0
    vpsrlw ymm9, ymm8, 4
    vpand ymm9, ymm9, ymm1
    vpsllw ymm9, ymm9, 4
    vpor ymm6, ymm6, ymm9
    vmovdqa ymm9, ymmword ptr [rdx + 192]
    vpmaddubsw ymm6, ymm6, ymm9
    vpmaddwd ymm6, ymm6, ymm2
    vcvtdq2ps ymm6, ymm6
    vmovd xmm9, dword ptr [rdi + 204]
    vpshufb xmm9, xmm9, xmmword ptr [rip + {c3}]
    vpmovsxbd ymm9, xmm9
    vcvtdq2ps ymm9, ymm9
    vmulps ymm9, ymm9, ymm5
    vbroadcastss ymm10, dword ptr [rcx + 24]
    vmulps ymm9, ymm9, ymm10
    vfmadd231ps ymm3, ymm6, ymm9
    vpsrlw ymm6, ymm7, 4
    vpand ymm6, ymm6, ymm0
    vpsrlw ymm7, ymm8, 6
    vpand ymm7, ymm7, ymm1
    vpsllw ymm7, ymm7, 4
    vpor ymm6, ymm6, ymm7
    vmovdqa ymm7, ymmword ptr [rdx + 224]
    vpmaddubsw ymm6, ymm6, ymm7
    vpmaddwd ymm6, ymm6, ymm2
    vcvtdq2ps ymm6, ymm6
    vmovd xmm7, dword ptr [rdi + 206]
    vpshufb xmm7, xmm7, xmmword ptr [rip + {c3}]
    vpmovsxbd ymm7, xmm7
    vcvtdq2ps ymm7, ymm7
    vmulps ymm7, ymm7, ymm5
    vbroadcastss ymm8, dword ptr [rcx + 28]
    vmulps ymm7, ymm7, ymm8
    vfmadd231ps ymm3, ymm6, ymm7
    vmovq xmm6, qword ptr [rdi + 192]
    vpmovsxbd ymm6, xmm6
    vcvtdq2ps ymm6, ymm6
    vmulps ymm6, ymm6, ymm5
    vmulps ymm6, ymm6, ymmword ptr [rip + {c4}]
    vmovups ymm7, ymmword ptr [r8]
    vfmadd231ps ymm4, ymm6, ymm7
    vmovq xmm6, qword ptr [rdi + 200]
    vpmovsxbd ymm6, xmm6
    vcvtdq2ps ymm6, ymm6
    vmulps ymm6, ymm6, ymm5
    vmulps ymm6, ymm6, ymmword ptr [rip + {c4}]
    vmovups ymm5, ymmword ptr [r8 + 32]
    vfmadd231ps ymm4, ymm6, ymm5
    add rdi, 210
    add rdx, 256
    add rcx, 32
    add r8, 64
    dec rsi
    jnz 2b
    vaddps ymm3, ymm3, ymm4
    vextractf128 xmm0, ymm3, 1
    vaddps xmm0, xmm3, xmm0
    vmovhlps xmm1, xmm0, xmm0
    vaddps xmm0, xmm0, xmm1
    vmovshdup xmm1, xmm0
    vaddss xmm0, xmm0, xmm1
    vzeroupper
    ret
3:
    vxorps xmm0, xmm0, xmm0
    vzeroupper
    ret
# c0, aligned to 4:
#   0f 0f 0f 0f
# c1, aligned to 4:
#   03 03 03 03
# c2, aligned to 4:
#   01 00 01 00
# c3, aligned to 16:
#   00 00 00 00 01 01 01 01 80 80 80 80 80 80 80 80
# c4, aligned to 32:
#   00 00 00 c2 00 00 00 c2 00 00 00 c2 00 00 00 c2
#   00 00 00 c2 00 00 00 c2 00 00 00 c2 00 00 00 c2
