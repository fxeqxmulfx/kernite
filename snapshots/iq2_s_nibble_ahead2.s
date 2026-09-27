# iq2_s_nibble_ahead2(gpr, gpr, gpr, gpr) -> vec
# frame: 128 bytes aligned to 128, saves []
# row=rdi superblocks=rsi quants=rdx scales=rcx levels=ymm0 one16=ymm1
# nibble=ymm2 acc=ymm3 cur=rax nxt=r8 p0=ymm4 p1=ymm5
# p2=ymm8 p3=ymm9 p4=ymm5 p5=ymm8 p6=ymm6 p7=ymm9
# d=ymm4,ymm6 steps=ymm5,ymm7 low_scales=ymm6,ymm4 high_scales=ymm5,ymm7 packed=ymm4,ymm5,ymm8,ymm9,ymm6 low_codes=ymm5,ymm6,ymm9,ymm10,ymm7
# unsigned_levels=ymm5,ymm6,ymm9,ymm10,ymm4,ymm0 x=ymm6,ymm7,ymm10,ymm11,ymm5,ymm2 lanes=ymm4,ymm5,ymm8,ymm9,ymm6,ymm0 low_scale=ymm9,ymm10,ymm5,ymm6,ymm7,ymm2,ymm1 high_scale=ymm10,ymm11,ymm8,ymm7,ymm6,ymm4,ymm2 scale=ymm9,ymm10,ymm5,ymm6,ymm7,ymm2,ymm1
# zero=ymm0 high=ymm0 sum=ymm0 pair=ymm1 result=ymm0
    push rbp
    mov rbp, rsp
    and rsp, -128
    sub rsp, 128
    test rsi, rsi
    jz 4f
    vmovdqa ymm0, ymmword ptr [rip + {c0}]
    vpbroadcastd ymm1, dword ptr [rip + {c1}]
    vpbroadcastd ymm2, dword ptr [rip + {c2}]
    vxorps xmm3, xmm3, xmm3
    lea rax, [rsp]
    lea r8, [rsp + 64]
    vpbroadcastw xmm4, word ptr [rdi]
    vcvtph2ps ymm4, xmm4
    vpmovzxbd ymm5, qword ptr [rdi + 2]
    vpand ymm6, ymm5, ymmword ptr [rip + {c3}]
    vpsrld ymm5, ymm5, 4
    vpaddd ymm6, ymm6, ymm6
    vpor ymm6, ymm6, ymmword ptr [rip + {c4}]
    vcvtdq2ps ymm6, ymm6
    vmulps ymm6, ymm4, ymm6
    vmulps ymm6, ymm6, ymmword ptr [rip + {c5}]
    vmulps ymm6, ymm6, ymmword ptr [rcx]
    vpaddd ymm5, ymm5, ymm5
    vpor ymm5, ymm5, ymmword ptr [rip + {c4}]
    vcvtdq2ps ymm5, ymm5
    vmulps ymm5, ymm4, ymm5
    vmulps ymm5, ymm5, ymmword ptr [rip + {c5}]
    vmulps ymm5, ymm5, ymmword ptr [rcx]
    vmovaps ymmword ptr [rax], ymm6
    vmovaps ymmword ptr [rax + 32], ymm5
    vmovdqu xmm4, xmmword ptr [rdi + 10]
    vpand xmm5, xmm4, xmm2
    vpsrlw xmm4, xmm4, 4
    vpand xmm4, xmm4, xmm2
    vinserti128 ymm4, ymm5, xmm4, 1
    vpshufb ymm4, ymm0, ymm4
    vpabsb ymm5, ymm4
    vmovdqa ymm6, ymmword ptr [rdx]
    vpsignb ymm6, ymm6, ymm4
    vpmaddubsw ymm4, ymm5, ymm6
    vmovdqu xmm5, xmmword ptr [rdi + 26]
    vpand xmm6, xmm5, xmm2
    vpsrlw xmm5, xmm5, 4
    vpand xmm5, xmm5, xmm2
    vinserti128 ymm5, ymm6, xmm5, 1
    vpshufb ymm5, ymm0, ymm5
    vpabsb ymm6, ymm5
    vmovdqa ymm7, ymmword ptr [rdx + 32]
    vpsignb ymm7, ymm7, ymm5
    vpmaddubsw ymm5, ymm6, ymm7
    dec rsi
    jz 3f
2:
    vpbroadcastw xmm6, word ptr [rdi + 138]
    vcvtph2ps ymm6, xmm6
    vpmovzxbd ymm7, qword ptr [rdi + 140]
    vmovdqu xmm8, xmmword ptr [rdi + 42]
    vpand xmm9, xmm8, xmm2
    vpsrlw xmm8, xmm8, 4
    vpand xmm8, xmm8, xmm2
    vinserti128 ymm8, ymm9, xmm8, 1
    vpshufb ymm8, ymm0, ymm8
    vpabsb ymm9, ymm8
    vmovdqa ymm10, ymmword ptr [rdx + 64]
    vpsignb ymm10, ymm10, ymm8
    vpmaddubsw ymm8, ymm9, ymm10
    vpmaddwd ymm4, ymm4, ymm1
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm9, dword ptr [rax]
    vbroadcastss ymm10, dword ptr [rax + 32]
    vpblendd ymm9, ymm9, ymm10, 0xf0
    vfmadd231ps ymm3, ymm4, ymm9
    vpand ymm4, ymm7, ymmword ptr [rip + {c3}]
    vpsrld ymm7, ymm7, 4
    vpaddd ymm4, ymm4, ymm4
    vmovdqu xmm9, xmmword ptr [rdi + 58]
    vpand xmm10, xmm9, xmm2
    vpsrlw xmm9, xmm9, 4
    vpand xmm9, xmm9, xmm2
    vinserti128 ymm9, ymm10, xmm9, 1
    vpshufb ymm9, ymm0, ymm9
    vpabsb ymm10, ymm9
    vmovdqa ymm11, ymmword ptr [rdx + 96]
    vpsignb ymm11, ymm11, ymm9
    vpmaddubsw ymm9, ymm10, ymm11
    vpmaddwd ymm5, ymm5, ymm1
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm10, dword ptr [rax + 4]
    vbroadcastss ymm11, dword ptr [rax + 36]
    vpblendd ymm10, ymm10, ymm11, 0xf0
    vfmadd231ps ymm3, ymm5, ymm10
    vpor ymm4, ymm4, ymmword ptr [rip + {c4}]
    vcvtdq2ps ymm4, ymm4
    vmulps ymm4, ymm6, ymm4
    vmovdqu xmm5, xmmword ptr [rdi + 74]
    vpand xmm10, xmm5, xmm2
    vpsrlw xmm5, xmm5, 4
    vpand xmm5, xmm5, xmm2
    vinserti128 ymm5, ymm10, xmm5, 1
    vpshufb ymm5, ymm0, ymm5
    vpabsb ymm10, ymm5
    vmovdqa ymm11, ymmword ptr [rdx + 128]
    vpsignb ymm11, ymm11, ymm5
    vpmaddubsw ymm5, ymm10, ymm11
    vpmaddwd ymm8, ymm8, ymm1
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm10, dword ptr [rax + 8]
    vbroadcastss ymm11, dword ptr [rax + 40]
    vpblendd ymm10, ymm10, ymm11, 0xf0
    vfmadd231ps ymm3, ymm8, ymm10
    vmulps ymm4, ymm4, ymmword ptr [rip + {c5}]
    vmulps ymm4, ymm4, ymmword ptr [rcx + 32]
    vpaddd ymm7, ymm7, ymm7
    vmovdqu xmm8, xmmword ptr [rdi + 90]
    vpand xmm10, xmm8, xmm2
    vpsrlw xmm8, xmm8, 4
    vpand xmm8, xmm8, xmm2
    vinserti128 ymm8, ymm10, xmm8, 1
    vpshufb ymm8, ymm0, ymm8
    vpabsb ymm10, ymm8
    vmovdqa ymm11, ymmword ptr [rdx + 160]
    vpsignb ymm11, ymm11, ymm8
    vpmaddubsw ymm8, ymm10, ymm11
    vpmaddwd ymm9, ymm9, ymm1
    vcvtdq2ps ymm9, ymm9
    vbroadcastss ymm10, dword ptr [rax + 12]
    vbroadcastss ymm11, dword ptr [rax + 44]
    vpblendd ymm10, ymm10, ymm11, 0xf0
    vfmadd231ps ymm3, ymm9, ymm10
    vpor ymm7, ymm7, ymmword ptr [rip + {c4}]
    vcvtdq2ps ymm7, ymm7
    vmulps ymm7, ymm6, ymm7
    vmovdqu xmm6, xmmword ptr [rdi + 106]
    vpand xmm9, xmm6, xmm2
    vpsrlw xmm6, xmm6, 4
    vpand xmm6, xmm6, xmm2
    vinserti128 ymm6, ymm9, xmm6, 1
    vpshufb ymm6, ymm0, ymm6
    vpabsb ymm9, ymm6
    vmovdqa ymm10, ymmword ptr [rdx + 192]
    vpsignb ymm10, ymm10, ymm6
    vpmaddubsw ymm6, ymm9, ymm10
    vpmaddwd ymm5, ymm5, ymm1
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm9, dword ptr [rax + 16]
    vbroadcastss ymm10, dword ptr [rax + 48]
    vpblendd ymm9, ymm9, ymm10, 0xf0
    vfmadd231ps ymm3, ymm5, ymm9
    vmulps ymm7, ymm7, ymmword ptr [rip + {c5}]
    vmulps ymm7, ymm7, ymmword ptr [rcx + 32]
    vmovaps ymmword ptr [r8], ymm4
    vmovdqu xmm4, xmmword ptr [rdi + 122]
    vpand xmm5, xmm4, xmm2
    vpsrlw xmm4, xmm4, 4
    vpand xmm4, xmm4, xmm2
    vinserti128 ymm9, ymm5, xmm4, 1
    vpshufb ymm9, ymm0, ymm9
    vpabsb ymm4, ymm9
    vmovdqa ymm5, ymmword ptr [rdx + 224]
    vpsignb ymm5, ymm5, ymm9
    vpmaddubsw ymm9, ymm4, ymm5
    vpmaddwd ymm4, ymm8, ymm1
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, dword ptr [rax + 20]
    vbroadcastss ymm8, dword ptr [rax + 52]
    vpblendd ymm5, ymm5, ymm8, 0xf0
    vfmadd231ps ymm3, ymm4, ymm5
    vmovaps ymmword ptr [r8 + 32], ymm7
    vmovdqu xmm4, xmmword ptr [rdi + 148]
    vpand xmm5, xmm4, xmm2
    vpsrlw xmm4, xmm4, 4
    vpand xmm4, xmm4, xmm2
    vinserti128 ymm4, ymm5, xmm4, 1
    vpshufb ymm4, ymm0, ymm4
    vpabsb ymm5, ymm4
    vmovdqa ymm7, ymmword ptr [rdx + 256]
    vpsignb ymm7, ymm7, ymm4
    vpmaddubsw ymm4, ymm5, ymm7
    vpmaddwd ymm5, ymm6, ymm1
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 24]
    vbroadcastss ymm7, dword ptr [rax + 56]
    vpblendd ymm6, ymm6, ymm7, 0xf0
    vfmadd231ps ymm3, ymm5, ymm6
    vmovdqu xmm5, xmmword ptr [rdi + 164]
    vpand xmm6, xmm5, xmm2
    vpsrlw xmm5, xmm5, 4
    vpand xmm5, xmm5, xmm2
    vinserti128 ymm5, ymm6, xmm5, 1
    vpshufb ymm5, ymm0, ymm5
    vpabsb ymm6, ymm5
    vmovdqa ymm7, ymmword ptr [rdx + 288]
    vpsignb ymm7, ymm7, ymm5
    vpmaddubsw ymm5, ymm6, ymm7
    vpmaddwd ymm6, ymm9, ymm1
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 28]
    vbroadcastss ymm8, dword ptr [rax + 60]
    vpblendd ymm7, ymm7, ymm8, 0xf0
    vfmadd231ps ymm3, ymm6, ymm7
    add rdi, 138
    add rdx, 256
    add rcx, 32
    xor rax, 64
    xor r8, 64
    dec rsi
    jnz 2b
3:
    vmovdqu xmm6, xmmword ptr [rdi + 42]
    vpand xmm7, xmm6, xmm2
    vpsrlw xmm6, xmm6, 4
    vpand xmm6, xmm6, xmm2
    vinserti128 ymm8, ymm7, xmm6, 1
    vpshufb ymm8, ymm0, ymm8
    vpabsb ymm6, ymm8
    vmovdqa ymm7, ymmword ptr [rdx + 64]
    vpsignb ymm7, ymm7, ymm8
    vpmaddubsw ymm8, ymm6, ymm7
    vpmaddwd ymm4, ymm4, ymm1
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm6, dword ptr [rax]
    vbroadcastss ymm7, dword ptr [rax + 32]
    vpblendd ymm6, ymm6, ymm7, 0xf0
    vfmadd231ps ymm3, ymm4, ymm6
    vmovdqu xmm4, xmmword ptr [rdi + 58]
    vpand xmm6, xmm4, xmm2
    vpsrlw xmm4, xmm4, 4
    vpand xmm4, xmm4, xmm2
    vinserti128 ymm9, ymm6, xmm4, 1
    vpshufb ymm9, ymm0, ymm9
    vpabsb ymm4, ymm9
    vmovdqa ymm6, ymmword ptr [rdx + 96]
    vpsignb ymm6, ymm6, ymm9
    vpmaddubsw ymm9, ymm4, ymm6
    vpmaddwd ymm4, ymm5, ymm1
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, dword ptr [rax + 4]
    vbroadcastss ymm6, dword ptr [rax + 36]
    vpblendd ymm5, ymm5, ymm6, 0xf0
    vfmadd231ps ymm3, ymm4, ymm5
    vmovdqu xmm4, xmmword ptr [rdi + 74]
    vpand xmm5, xmm4, xmm2
    vpsrlw xmm4, xmm4, 4
    vpand xmm4, xmm4, xmm2
    vinserti128 ymm5, ymm5, xmm4, 1
    vpshufb ymm5, ymm0, ymm5
    vpabsb ymm4, ymm5
    vmovdqa ymm6, ymmword ptr [rdx + 128]
    vpsignb ymm6, ymm6, ymm5
    vpmaddubsw ymm5, ymm4, ymm6
    vpmaddwd ymm4, ymm8, ymm1
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm6, dword ptr [rax + 8]
    vbroadcastss ymm7, dword ptr [rax + 40]
    vpblendd ymm6, ymm6, ymm7, 0xf0
    vfmadd231ps ymm3, ymm4, ymm6
    vmovdqu xmm4, xmmword ptr [rdi + 90]
    vpand xmm6, xmm4, xmm2
    vpsrlw xmm4, xmm4, 4
    vpand xmm4, xmm4, xmm2
    vinserti128 ymm8, ymm6, xmm4, 1
    vpshufb ymm8, ymm0, ymm8
    vpabsb ymm4, ymm8
    vmovdqa ymm6, ymmword ptr [rdx + 160]
    vpsignb ymm6, ymm6, ymm8
    vpmaddubsw ymm8, ymm4, ymm6
    vpmaddwd ymm4, ymm9, ymm1
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm6, dword ptr [rax + 12]
    vbroadcastss ymm7, dword ptr [rax + 44]
    vpblendd ymm6, ymm6, ymm7, 0xf0
    vfmadd231ps ymm3, ymm4, ymm6
    vmovdqu xmm4, xmmword ptr [rdi + 106]
    vpand xmm6, xmm4, xmm2
    vpsrlw xmm4, xmm4, 4
    vpand xmm4, xmm4, xmm2
    vinserti128 ymm6, ymm6, xmm4, 1
    vpshufb ymm6, ymm0, ymm6
    vpabsb ymm4, ymm6
    vmovdqa ymm7, ymmword ptr [rdx + 192]
    vpsignb ymm7, ymm7, ymm6
    vpmaddubsw ymm6, ymm4, ymm7
    vpmaddwd ymm4, ymm5, ymm1
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, dword ptr [rax + 16]
    vbroadcastss ymm7, dword ptr [rax + 48]
    vpblendd ymm5, ymm5, ymm7, 0xf0
    vfmadd231ps ymm3, ymm4, ymm5
    vmovdqu xmm4, xmmword ptr [rdi + 122]
    vpand xmm5, xmm4, xmm2
    vpsrlw xmm4, xmm4, 4
    vpand xmm4, xmm4, xmm2
    vinserti128 ymm9, ymm5, xmm4, 1
    vpshufb ymm9, ymm0, ymm9
    vpabsb ymm0, ymm9
    vmovdqa ymm2, ymmword ptr [rdx + 224]
    vpsignb ymm2, ymm2, ymm9
    vpmaddubsw ymm9, ymm0, ymm2
    vpmaddwd ymm0, ymm8, ymm1
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm2, dword ptr [rax + 20]
    vbroadcastss ymm4, dword ptr [rax + 52]
    vpblendd ymm2, ymm2, ymm4, 0xf0
    vfmadd231ps ymm3, ymm0, ymm2
    vpmaddwd ymm0, ymm6, ymm1
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm2, dword ptr [rax + 24]
    vbroadcastss ymm4, dword ptr [rax + 56]
    vpblendd ymm2, ymm2, ymm4, 0xf0
    vfmadd231ps ymm3, ymm0, ymm2
    vpmaddwd ymm0, ymm9, ymm1
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm1, dword ptr [rax + 28]
    vbroadcastss ymm2, dword ptr [rax + 60]
    vpblendd ymm1, ymm1, ymm2, 0xf0
    vfmadd231ps ymm3, ymm0, ymm1
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
#   08 19 2b 00 00 00 00 00 f8 e7 d5 00 00 00 00 00
#   08 19 2b 00 00 00 00 00 f8 e7 d5 00 00 00 00 00
# c1, aligned to 4:
#   01 00 01 00
# c2, aligned to 4:
#   0f 0f 0f 0f
# c3, aligned to 32:
#   0f 00 00 00 0f 00 00 00 0f 00 00 00 0f 00 00 00
#   0f 00 00 00 0f 00 00 00 0f 00 00 00 0f 00 00 00
# c4, aligned to 32:
#   01 00 00 00 01 00 00 00 01 00 00 00 01 00 00 00
#   01 00 00 00 01 00 00 00 01 00 00 00 01 00 00 00
# c5, aligned to 32:
#   00 00 00 3e 00 00 00 3e 00 00 00 3e 00 00 00 3e
#   00 00 00 3e 00 00 00 3e 00 00 00 3e 00 00 00 3e
