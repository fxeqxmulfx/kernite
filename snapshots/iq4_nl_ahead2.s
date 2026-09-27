# iq4_nl_ahead2(gpr, gpr, gpr, gpr) -> vec
# frame: 0 bytes aligned to 0, saves []
# row=rdi groups=rsi quants=rdx scales=rcx levels=ymm0 nibble=ymm1
# one16=ymm2 acc=ymm3 p0=ymm4 p1=ymm5 p2=ymm6 p3=ymm7
# packed=ymm4,ymm5,ymm6 low_codes=ymm5,ymm6,ymm7 signed_levels=ymm4,ymm5,ymm6,ymm1 unsigned_levels=ymm5,ymm6,ymm7,ymm0 x=ymm6,ymm7,ymm8,ymm4 lanes=ymm4,ymm5,ymm6,ymm0
# d=ymm7,ymm5,ymm6,ymm1 dx=ymm8,ymm4,ymm2 zero=ymm0 high=ymm0 sum=ymm0 pair=ymm1
# result=ymm0 nothing=ymm0
    test rsi, rsi
    jz 4f
    vmovdqa ymm0, ymmword ptr [rip + {c0}]
    vpbroadcastd ymm1, dword ptr [rip + {c1}]
    vpbroadcastd ymm2, dword ptr [rip + {c2}]
    vxorps xmm3, xmm3, xmm3
    # products of block 0 of group +0
    vmovdqu xmm4, xmmword ptr [rdi + 2]
    vpand xmm5, xmm4, xmm1
    vpsrlw xmm4, xmm4, 4
    vpand xmm4, xmm4, xmm1
    vinserti128 ymm4, ymm5, xmm4, 1
    vpshufb ymm4, ymm0, ymm4
    vpabsb ymm5, ymm4
    vmovdqa ymm6, ymmword ptr [rdx]
    vpsignb ymm6, ymm6, ymm4
    vpmaddubsw ymm4, ymm5, ymm6
    # products of block 1 of group +0
    vmovdqu xmm5, xmmword ptr [rdi + 20]
    vpand xmm6, xmm5, xmm1
    vpsrlw xmm5, xmm5, 4
    vpand xmm5, xmm5, xmm1
    vinserti128 ymm5, ymm6, xmm5, 1
    vpshufb ymm5, ymm0, ymm5
    vpabsb ymm6, ymm5
    vmovdqa ymm7, ymmword ptr [rdx + 32]
    vpsignb ymm7, ymm7, ymm5
    vpmaddubsw ymm5, ymm6, ymm7
    dec rsi
    jz 3f
2:
    # products of block 2 of group +0
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
    # sum of block 0
    vpmaddwd ymm4, ymm4, ymm2
    vcvtdq2ps ymm4, ymm4
    vpbroadcastw xmm7, word ptr [rdi]
    vcvtph2ps ymm7, xmm7
    vbroadcastss ymm8, dword ptr [rcx]
    vmulps ymm7, ymm7, ymm8
    vfmadd231ps ymm3, ymm4, ymm7
    # products of block 3 of group +0
    vmovdqu xmm4, xmmword ptr [rdi + 56]
    vpand xmm7, xmm4, xmm1
    vpsrlw xmm4, xmm4, 4
    vpand xmm4, xmm4, xmm1
    vinserti128 ymm4, ymm7, xmm4, 1
    vpshufb ymm4, ymm0, ymm4
    vpabsb ymm7, ymm4
    vmovdqa ymm8, ymmword ptr [rdx + 96]
    vpsignb ymm8, ymm8, ymm4
    vpmaddubsw ymm7, ymm7, ymm8
    # sum of block 1
    vpmaddwd ymm4, ymm5, ymm2
    vcvtdq2ps ymm4, ymm4
    vpbroadcastw xmm5, word ptr [rdi + 18]
    vcvtph2ps ymm5, xmm5
    vbroadcastss ymm8, dword ptr [rcx + 4]
    vmulps ymm5, ymm5, ymm8
    vfmadd231ps ymm3, ymm4, ymm5
    # products of block 0 of group +1
    vmovdqu xmm4, xmmword ptr [rdi + 74]
    vpand xmm5, xmm4, xmm1
    vpsrlw xmm4, xmm4, 4
    vpand xmm4, xmm4, xmm1
    vinserti128 ymm4, ymm5, xmm4, 1
    vpshufb ymm4, ymm0, ymm4
    vpabsb ymm5, ymm4
    vmovdqa ymm8, ymmword ptr [rdx + 128]
    vpsignb ymm8, ymm8, ymm4
    vpmaddubsw ymm4, ymm5, ymm8
    # sum of block 2
    vpmaddwd ymm5, ymm6, ymm2
    vcvtdq2ps ymm5, ymm5
    vpbroadcastw xmm6, word ptr [rdi + 36]
    vcvtph2ps ymm6, xmm6
    vbroadcastss ymm8, dword ptr [rcx + 8]
    vmulps ymm6, ymm6, ymm8
    vfmadd231ps ymm3, ymm5, ymm6
    # products of block 1 of group +1
    vmovdqu xmm5, xmmword ptr [rdi + 92]
    vpand xmm6, xmm5, xmm1
    vpsrlw xmm5, xmm5, 4
    vpand xmm5, xmm5, xmm1
    vinserti128 ymm5, ymm6, xmm5, 1
    vpshufb ymm5, ymm0, ymm5
    vpabsb ymm6, ymm5
    vmovdqa ymm8, ymmword ptr [rdx + 160]
    vpsignb ymm8, ymm8, ymm5
    vpmaddubsw ymm5, ymm6, ymm8
    # sum of block 3
    vpmaddwd ymm6, ymm7, ymm2
    vcvtdq2ps ymm6, ymm6
    vpbroadcastw xmm7, word ptr [rdi + 54]
    vcvtph2ps ymm7, xmm7
    vbroadcastss ymm8, dword ptr [rcx + 12]
    vmulps ymm7, ymm7, ymm8
    vfmadd231ps ymm3, ymm6, ymm7
    add rdi, 72
    add rdx, 128
    add rcx, 16
    dec rsi
    jnz 2b
3:
    # products of block 2 of group +0
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
    # sum of block 0
    vpmaddwd ymm4, ymm4, ymm2
    vcvtdq2ps ymm4, ymm4
    vpbroadcastw xmm7, word ptr [rdi]
    vcvtph2ps ymm7, xmm7
    vbroadcastss ymm8, dword ptr [rcx]
    vmulps ymm7, ymm7, ymm8
    vfmadd231ps ymm3, ymm4, ymm7
    # products of block 3 of group +0
    vmovdqu xmm4, xmmword ptr [rdi + 56]
    vpand xmm7, xmm4, xmm1
    vpsrlw xmm4, xmm4, 4
    vpand xmm4, xmm4, xmm1
    vinserti128 ymm1, ymm7, xmm4, 1
    vpshufb ymm1, ymm0, ymm1
    vpabsb ymm0, ymm1
    vmovdqa ymm4, ymmword ptr [rdx + 96]
    vpsignb ymm4, ymm4, ymm1
    vpmaddubsw ymm7, ymm0, ymm4
    # sum of block 1
    vpmaddwd ymm0, ymm5, ymm2
    vcvtdq2ps ymm0, ymm0
    vpbroadcastw xmm1, word ptr [rdi + 18]
    vcvtph2ps ymm1, xmm1
    vbroadcastss ymm4, dword ptr [rcx + 4]
    vmulps ymm1, ymm1, ymm4
    vfmadd231ps ymm3, ymm0, ymm1
    # sum of block 2
    vpmaddwd ymm0, ymm6, ymm2
    vcvtdq2ps ymm0, ymm0
    vpbroadcastw xmm1, word ptr [rdi + 36]
    vcvtph2ps ymm1, xmm1
    vbroadcastss ymm4, dword ptr [rcx + 8]
    vmulps ymm1, ymm1, ymm4
    vfmadd231ps ymm3, ymm0, ymm1
    # sum of block 3
    vpmaddwd ymm0, ymm7, ymm2
    vcvtdq2ps ymm0, ymm0
    vpbroadcastw xmm1, word ptr [rdi + 54]
    vcvtph2ps ymm1, xmm1
    vbroadcastss ymm2, dword ptr [rcx + 12]
    vmulps ymm1, ymm1, ymm2
    vfmadd231ps ymm3, ymm0, ymm1
    vxorps xmm0, xmm0, xmm0
    vaddps ymm3, ymm3, ymm0
    vextractf128 xmm0, ymm3, 1
    vaddps xmm0, xmm3, xmm0
    vmovhlps xmm1, xmm0, xmm0
    vaddps xmm0, xmm0, xmm1
    vmovshdup xmm1, xmm0
    vaddss xmm0, xmm0, xmm1
    vzeroupper
    ret
4:
    vxorps xmm0, xmm0, xmm0
    vzeroupper
    ret
# c0, aligned to 32:
#   81 98 ad bf cf dd ea f6 01 0d 19 26 35 45 59 71
#   81 98 ad bf cf dd ea f6 01 0d 19 26 35 45 59 71
# c1, aligned to 4:
#   0f 0f 0f 0f
# c2, aligned to 4:
#   01 00 01 00
