# iq2_s_planes_ahead2(gpr, gpr, gpr, gpr) -> vec
# frame: 128 bytes aligned to 128, saves []
# row=rdi superblocks=rsi quants=rdx scales=rcx levels=ymm0 one16=ymm1
# three=ymm2 spread=ymm3 bit=ymm4 max8=ymm5 acc=ymm6 cur=rax
# nxt=r8 p0=ymm7 p1=ymm8 p2=ymm11 p3=ymm12 p4=ymm8
# p5=ymm11 p6=ymm9 p7=ymm12 d=ymm7,ymm9 steps=ymm8,ymm10 low_scales=ymm9,ymm7
# high_scales=ymm8,ymm10 codes=ymm7,ymm8,ymm11,ymm12,ymm9 signs=ymm8,ymm9,ymm12,ymm13,ymm10,ymm0 x=ymm9,ymm10,ymm13,ymm14,ymm12,ymm11,ymm2 lanes=ymm7,ymm8,ymm11,ymm12,ymm9,ymm0 low_scale=ymm12,ymm13,ymm8,ymm9,ymm10,ymm2,ymm1
# high_scale=ymm13,ymm14,ymm11,ymm10,ymm9,ymm3,ymm2 scale=ymm12,ymm13,ymm8,ymm9,ymm10,ymm2,ymm1 zero=ymm0 high=ymm0 sum=ymm0 pair=ymm1
# result=ymm0
    push rbp
    mov rbp, rsp
    and rsp, -128
    sub rsp, 128
    test rsi, rsi
    jz 4f
    vmovdqa ymm0, ymmword ptr [rip + {c0}]
    vpbroadcastd ymm1, dword ptr [rip + {c1}]
    vpbroadcastd ymm2, dword ptr [rip + {c2}]
    vmovdqa ymm3, ymmword ptr [rip + {c3}]
    vpbroadcastq ymm4, qword ptr [rip + {c4}]
    vpbroadcastd ymm5, dword ptr [rip + {c5}]
    vxorps xmm6, xmm6, xmm6
    lea rax, [rsp]
    lea r8, [rsp + 64]
    vpbroadcastw xmm7, word ptr [rdi]
    vcvtph2ps ymm7, xmm7
    vpmovzxbd ymm8, qword ptr [rdi + 2]
    vpand ymm9, ymm8, ymmword ptr [rip + {c6}]
    vpsrld ymm8, ymm8, 4
    vpaddd ymm9, ymm9, ymm9
    vpor ymm9, ymm9, ymmword ptr [rip + {c7}]
    vcvtdq2ps ymm9, ymm9
    vmulps ymm9, ymm7, ymm9
    vmulps ymm9, ymm9, ymmword ptr [rip + {c8}]
    vmulps ymm9, ymm9, ymmword ptr [rcx]
    vpaddd ymm8, ymm8, ymm8
    vpor ymm8, ymm8, ymmword ptr [rip + {c7}]
    vcvtdq2ps ymm8, ymm8
    vmulps ymm8, ymm7, ymm8
    vmulps ymm8, ymm8, ymmword ptr [rip + {c8}]
    vmulps ymm8, ymm8, ymmword ptr [rcx]
    vmovaps ymmword ptr [rax], ymm9
    vmovaps ymmword ptr [rax + 32], ymm8
    vmovdqu ymm7, ymmword ptr [rdi + 10]
    vpand ymm7, ymm7, ymm2
    vpshufb ymm7, ymm0, ymm7
    vpbroadcastd ymm8, dword ptr [rdi + 74]
    vpshufb ymm8, ymm8, ymm3
    vpand ymm8, ymm8, ymm4
    vpaddb ymm8, ymm8, ymm5
    vmovdqa ymm9, ymmword ptr [rdx]
    vpsignb ymm9, ymm9, ymm8
    vpmaddubsw ymm7, ymm7, ymm9
    vmovdqu ymm8, ymmword ptr [rdi + 10]
    vpsrlw ymm8, ymm8, 2
    vpand ymm8, ymm8, ymm2
    vpshufb ymm8, ymm0, ymm8
    vpbroadcastd ymm9, dword ptr [rdi + 78]
    vpshufb ymm9, ymm9, ymm3
    vpand ymm9, ymm9, ymm4
    vpaddb ymm9, ymm9, ymm5
    vmovdqa ymm10, ymmword ptr [rdx + 32]
    vpsignb ymm10, ymm10, ymm9
    vpmaddubsw ymm8, ymm8, ymm10
    dec rsi
    jz 3f
2:
    vpbroadcastw xmm9, word ptr [rdi + 106]
    vcvtph2ps ymm9, xmm9
    vpmovzxbd ymm10, qword ptr [rdi + 108]
    vmovdqu ymm11, ymmword ptr [rdi + 10]
    vpsrlw ymm11, ymm11, 4
    vpand ymm11, ymm11, ymm2
    vpshufb ymm11, ymm0, ymm11
    vpbroadcastd ymm12, dword ptr [rdi + 82]
    vpshufb ymm12, ymm12, ymm3
    vpand ymm12, ymm12, ymm4
    vpaddb ymm12, ymm12, ymm5
    vmovdqa ymm13, ymmword ptr [rdx + 64]
    vpsignb ymm13, ymm13, ymm12
    vpmaddubsw ymm11, ymm11, ymm13
    vpmaddwd ymm7, ymm7, ymm1
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm12, dword ptr [rax]
    vbroadcastss ymm13, dword ptr [rax + 32]
    vpblendd ymm12, ymm12, ymm13, 0xf0
    vfmadd231ps ymm6, ymm7, ymm12
    vpand ymm7, ymm10, ymmword ptr [rip + {c6}]
    vpsrld ymm10, ymm10, 4
    vpaddd ymm7, ymm7, ymm7
    vmovdqu ymm12, ymmword ptr [rdi + 10]
    vpsrlw ymm12, ymm12, 6
    vpand ymm12, ymm12, ymm2
    vpshufb ymm12, ymm0, ymm12
    vpbroadcastd ymm13, dword ptr [rdi + 86]
    vpshufb ymm13, ymm13, ymm3
    vpand ymm13, ymm13, ymm4
    vpaddb ymm13, ymm13, ymm5
    vmovdqa ymm14, ymmword ptr [rdx + 96]
    vpsignb ymm14, ymm14, ymm13
    vpmaddubsw ymm12, ymm12, ymm14
    vpmaddwd ymm8, ymm8, ymm1
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm13, dword ptr [rax + 4]
    vbroadcastss ymm14, dword ptr [rax + 36]
    vpblendd ymm13, ymm13, ymm14, 0xf0
    vfmadd231ps ymm6, ymm8, ymm13
    vpor ymm7, ymm7, ymmword ptr [rip + {c7}]
    vcvtdq2ps ymm7, ymm7
    vmulps ymm7, ymm9, ymm7
    vmovdqu ymm8, ymmword ptr [rdi + 42]
    vpand ymm8, ymm8, ymm2
    vpshufb ymm8, ymm0, ymm8
    vpbroadcastd ymm13, dword ptr [rdi + 90]
    vpshufb ymm13, ymm13, ymm3
    vpand ymm13, ymm13, ymm4
    vpaddb ymm13, ymm13, ymm5
    vmovdqa ymm14, ymmword ptr [rdx + 128]
    vpsignb ymm14, ymm14, ymm13
    vpmaddubsw ymm8, ymm8, ymm14
    vpmaddwd ymm11, ymm11, ymm1
    vcvtdq2ps ymm11, ymm11
    vbroadcastss ymm13, dword ptr [rax + 8]
    vbroadcastss ymm14, dword ptr [rax + 40]
    vpblendd ymm13, ymm13, ymm14, 0xf0
    vfmadd231ps ymm6, ymm11, ymm13
    vmulps ymm7, ymm7, ymmword ptr [rip + {c8}]
    vmulps ymm7, ymm7, ymmword ptr [rcx + 32]
    vpaddd ymm10, ymm10, ymm10
    vmovdqu ymm11, ymmword ptr [rdi + 42]
    vpsrlw ymm11, ymm11, 2
    vpand ymm11, ymm11, ymm2
    vpshufb ymm11, ymm0, ymm11
    vpbroadcastd ymm13, dword ptr [rdi + 94]
    vpshufb ymm13, ymm13, ymm3
    vpand ymm13, ymm13, ymm4
    vpaddb ymm13, ymm13, ymm5
    vmovdqa ymm14, ymmword ptr [rdx + 160]
    vpsignb ymm14, ymm14, ymm13
    vpmaddubsw ymm11, ymm11, ymm14
    vpmaddwd ymm12, ymm12, ymm1
    vcvtdq2ps ymm12, ymm12
    vbroadcastss ymm13, dword ptr [rax + 12]
    vbroadcastss ymm14, dword ptr [rax + 44]
    vpblendd ymm13, ymm13, ymm14, 0xf0
    vfmadd231ps ymm6, ymm12, ymm13
    vpor ymm10, ymm10, ymmword ptr [rip + {c7}]
    vcvtdq2ps ymm10, ymm10
    vmulps ymm10, ymm9, ymm10
    vmovdqu ymm9, ymmword ptr [rdi + 42]
    vpsrlw ymm9, ymm9, 4
    vpand ymm9, ymm9, ymm2
    vpshufb ymm9, ymm0, ymm9
    vpbroadcastd ymm12, dword ptr [rdi + 98]
    vpshufb ymm12, ymm12, ymm3
    vpand ymm12, ymm12, ymm4
    vpaddb ymm12, ymm12, ymm5
    vmovdqa ymm13, ymmword ptr [rdx + 192]
    vpsignb ymm13, ymm13, ymm12
    vpmaddubsw ymm9, ymm9, ymm13
    vpmaddwd ymm8, ymm8, ymm1
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm12, dword ptr [rax + 16]
    vbroadcastss ymm13, dword ptr [rax + 48]
    vpblendd ymm12, ymm12, ymm13, 0xf0
    vfmadd231ps ymm6, ymm8, ymm12
    vmulps ymm10, ymm10, ymmword ptr [rip + {c8}]
    vmulps ymm10, ymm10, ymmword ptr [rcx + 32]
    vmovaps ymmword ptr [r8], ymm7
    vmovdqu ymm7, ymmword ptr [rdi + 42]
    vpsrlw ymm7, ymm7, 6
    vpand ymm7, ymm7, ymm2
    vpshufb ymm7, ymm0, ymm7
    vpbroadcastd ymm8, dword ptr [rdi + 102]
    vpshufb ymm8, ymm8, ymm3
    vpand ymm8, ymm8, ymm4
    vpaddb ymm8, ymm8, ymm5
    vmovdqa ymm12, ymmword ptr [rdx + 224]
    vpsignb ymm12, ymm12, ymm8
    vpmaddubsw ymm12, ymm7, ymm12
    vpmaddwd ymm7, ymm11, ymm1
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rax + 20]
    vbroadcastss ymm11, dword ptr [rax + 52]
    vpblendd ymm8, ymm8, ymm11, 0xf0
    vfmadd231ps ymm6, ymm7, ymm8
    vmovaps ymmword ptr [r8 + 32], ymm10
    vmovdqu ymm7, ymmword ptr [rdi + 116]
    vpand ymm7, ymm7, ymm2
    vpshufb ymm7, ymm0, ymm7
    vpbroadcastd ymm8, dword ptr [rdi + 180]
    vpshufb ymm8, ymm8, ymm3
    vpand ymm8, ymm8, ymm4
    vpaddb ymm8, ymm8, ymm5
    vmovdqa ymm10, ymmword ptr [rdx + 256]
    vpsignb ymm10, ymm10, ymm8
    vpmaddubsw ymm7, ymm7, ymm10
    vpmaddwd ymm8, ymm9, ymm1
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm9, dword ptr [rax + 24]
    vbroadcastss ymm10, dword ptr [rax + 56]
    vpblendd ymm9, ymm9, ymm10, 0xf0
    vfmadd231ps ymm6, ymm8, ymm9
    vmovdqu ymm8, ymmword ptr [rdi + 116]
    vpsrlw ymm8, ymm8, 2
    vpand ymm8, ymm8, ymm2
    vpshufb ymm8, ymm0, ymm8
    vpbroadcastd ymm9, dword ptr [rdi + 184]
    vpshufb ymm9, ymm9, ymm3
    vpand ymm9, ymm9, ymm4
    vpaddb ymm9, ymm9, ymm5
    vmovdqa ymm10, ymmword ptr [rdx + 288]
    vpsignb ymm10, ymm10, ymm9
    vpmaddubsw ymm8, ymm8, ymm10
    vpmaddwd ymm9, ymm12, ymm1
    vcvtdq2ps ymm9, ymm9
    vbroadcastss ymm10, dword ptr [rax + 28]
    vbroadcastss ymm11, dword ptr [rax + 60]
    vpblendd ymm10, ymm10, ymm11, 0xf0
    vfmadd231ps ymm6, ymm9, ymm10
    add rdi, 106
    add rdx, 256
    add rcx, 32
    xor rax, 64
    xor r8, 64
    dec rsi
    jnz 2b
3:
    vmovdqu ymm9, ymmword ptr [rdi + 10]
    vpsrlw ymm9, ymm9, 4
    vpand ymm9, ymm9, ymm2
    vpshufb ymm9, ymm0, ymm9
    vpbroadcastd ymm10, dword ptr [rdi + 82]
    vpshufb ymm10, ymm10, ymm3
    vpand ymm10, ymm10, ymm4
    vpaddb ymm10, ymm10, ymm5
    vmovdqa ymm11, ymmword ptr [rdx + 64]
    vpsignb ymm11, ymm11, ymm10
    vpmaddubsw ymm11, ymm9, ymm11
    vpmaddwd ymm7, ymm7, ymm1
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm9, dword ptr [rax]
    vbroadcastss ymm10, dword ptr [rax + 32]
    vpblendd ymm9, ymm9, ymm10, 0xf0
    vfmadd231ps ymm6, ymm7, ymm9
    vmovdqu ymm7, ymmword ptr [rdi + 10]
    vpsrlw ymm7, ymm7, 6
    vpand ymm7, ymm7, ymm2
    vpshufb ymm7, ymm0, ymm7
    vpbroadcastd ymm9, dword ptr [rdi + 86]
    vpshufb ymm9, ymm9, ymm3
    vpand ymm9, ymm9, ymm4
    vpaddb ymm9, ymm9, ymm5
    vmovdqa ymm10, ymmword ptr [rdx + 96]
    vpsignb ymm10, ymm10, ymm9
    vpmaddubsw ymm12, ymm7, ymm10
    vpmaddwd ymm7, ymm8, ymm1
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rax + 4]
    vbroadcastss ymm9, dword ptr [rax + 36]
    vpblendd ymm8, ymm8, ymm9, 0xf0
    vfmadd231ps ymm6, ymm7, ymm8
    vmovdqu ymm7, ymmword ptr [rdi + 42]
    vpand ymm7, ymm7, ymm2
    vpshufb ymm7, ymm0, ymm7
    vpbroadcastd ymm8, dword ptr [rdi + 90]
    vpshufb ymm8, ymm8, ymm3
    vpand ymm8, ymm8, ymm4
    vpaddb ymm8, ymm8, ymm5
    vmovdqa ymm9, ymmword ptr [rdx + 128]
    vpsignb ymm9, ymm9, ymm8
    vpmaddubsw ymm8, ymm7, ymm9
    vpmaddwd ymm7, ymm11, ymm1
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm9, dword ptr [rax + 8]
    vbroadcastss ymm10, dword ptr [rax + 40]
    vpblendd ymm9, ymm9, ymm10, 0xf0
    vfmadd231ps ymm6, ymm7, ymm9
    vmovdqu ymm7, ymmword ptr [rdi + 42]
    vpsrlw ymm7, ymm7, 2
    vpand ymm7, ymm7, ymm2
    vpshufb ymm7, ymm0, ymm7
    vpbroadcastd ymm9, dword ptr [rdi + 94]
    vpshufb ymm9, ymm9, ymm3
    vpand ymm9, ymm9, ymm4
    vpaddb ymm9, ymm9, ymm5
    vmovdqa ymm10, ymmword ptr [rdx + 160]
    vpsignb ymm10, ymm10, ymm9
    vpmaddubsw ymm11, ymm7, ymm10
    vpmaddwd ymm7, ymm12, ymm1
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm9, dword ptr [rax + 12]
    vbroadcastss ymm10, dword ptr [rax + 44]
    vpblendd ymm9, ymm9, ymm10, 0xf0
    vfmadd231ps ymm6, ymm7, ymm9
    vmovdqu ymm7, ymmword ptr [rdi + 42]
    vpsrlw ymm7, ymm7, 4
    vpand ymm7, ymm7, ymm2
    vpshufb ymm7, ymm0, ymm7
    vpbroadcastd ymm9, dword ptr [rdi + 98]
    vpshufb ymm9, ymm9, ymm3
    vpand ymm9, ymm9, ymm4
    vpaddb ymm9, ymm9, ymm5
    vmovdqa ymm10, ymmword ptr [rdx + 192]
    vpsignb ymm10, ymm10, ymm9
    vpmaddubsw ymm9, ymm7, ymm10
    vpmaddwd ymm7, ymm8, ymm1
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rax + 16]
    vbroadcastss ymm10, dword ptr [rax + 48]
    vpblendd ymm8, ymm8, ymm10, 0xf0
    vfmadd231ps ymm6, ymm7, ymm8
    vmovdqu ymm7, ymmword ptr [rdi + 42]
    vpsrlw ymm7, ymm7, 6
    vpand ymm7, ymm7, ymm2
    vpshufb ymm7, ymm0, ymm7
    vpbroadcastd ymm0, dword ptr [rdi + 102]
    vpshufb ymm0, ymm0, ymm3
    vpand ymm0, ymm0, ymm4
    vpaddb ymm0, ymm0, ymm5
    vmovdqa ymm2, ymmword ptr [rdx + 224]
    vpsignb ymm2, ymm2, ymm0
    vpmaddubsw ymm12, ymm7, ymm2
    vpmaddwd ymm0, ymm11, ymm1
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm2, dword ptr [rax + 20]
    vbroadcastss ymm3, dword ptr [rax + 52]
    vpblendd ymm2, ymm2, ymm3, 0xf0
    vfmadd231ps ymm6, ymm0, ymm2
    vpmaddwd ymm0, ymm9, ymm1
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm2, dword ptr [rax + 24]
    vbroadcastss ymm3, dword ptr [rax + 56]
    vpblendd ymm2, ymm2, ymm3, 0xf0
    vfmadd231ps ymm6, ymm0, ymm2
    vpmaddwd ymm0, ymm12, ymm1
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm1, dword ptr [rax + 28]
    vbroadcastss ymm2, dword ptr [rax + 60]
    vpblendd ymm1, ymm1, ymm2, 0xf0
    vfmadd231ps ymm6, ymm0, ymm1
    vxorps xmm0, xmm0, xmm0
    vaddps ymm6, ymm6, ymm0
    vextractf128 xmm0, ymm6, 1
    vaddps xmm0, xmm6, xmm0
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
#   08 19 2b 00 08 19 2b 00 08 19 2b 00 08 19 2b 00
#   08 19 2b 00 08 19 2b 00 08 19 2b 00 08 19 2b 00
# c1, aligned to 4:
#   01 00 01 00
# c2, aligned to 4:
#   03 03 03 03
# c3, aligned to 32:
#   00 00 00 00 00 00 00 00 01 01 01 01 01 01 01 01
#   02 02 02 02 02 02 02 02 03 03 03 03 03 03 03 03
# c4, aligned to 8:
#   01 02 04 08 10 20 40 80
# c5, aligned to 4:
#   7f 7f 7f 7f
# c6, aligned to 32:
#   0f 00 00 00 0f 00 00 00 0f 00 00 00 0f 00 00 00
#   0f 00 00 00 0f 00 00 00 0f 00 00 00 0f 00 00 00
# c7, aligned to 32:
#   01 00 00 00 01 00 00 00 01 00 00 00 01 00 00 00
#   01 00 00 00 01 00 00 00 01 00 00 00 01 00 00 00
# c8, aligned to 32:
#   00 00 00 3e 00 00 00 3e 00 00 00 3e 00 00 00 3e
#   00 00 00 3e 00 00 00 3e 00 00 00 3e 00 00 00 3e
