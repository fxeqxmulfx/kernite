# q4_k_x2_ahead1024_t1(gpr, gpr, gpr, gpr, gpr, gpr) -> ()
# frame: 1024 bytes aligned to 512, saves []
# row=rdi sbs=rsi quants=rdx scales=rcx half_sums=r8 out=r9
# nibble_mask=ymm0 one16=ymm1 stride=rax sums1=rax acc0=ymm2 acc1=ymm3
# zero=ymm3,ymm4 cur=r10 nxt=r11 packed=ymm4,ymm6 low_six=ymm5,ymm7 top_bits=ymm4,ymm6
# last_four=ymm6,ymm5 blocks_0_3=ymm4,ymm6 blocks_4_7=ymm5,ymm7 d_dmin=ymm6,ymm5 d_and_minus_dmin=ymm6,ymm5 scale_products=ymm4,ymm5
# offsets_lo=ymm4 offsets_hi=ymm5 off=ymm6 codes=ymm4 low_nibbles=ymm5,ymm8,ymm6 high_nibbles=ymm4
# products=ymm8,ymm5,ymm4,ymm6,ymm0 block_scale=ymm9,ymm8,ymm5,ymm6,ymm7,ymm1 sum=ymm0 high=ymm1 pair=ymm1 result=ymm0
# nothing=ymm0
    push rbp
    mov rbp, rsp
    and rsp, -512
    sub rsp, 1024
    test rsi, rsi
    jz 4f
    # constants
    vpbroadcastd ymm0, dword ptr [rip + {c0}]
    vpbroadcastd ymm1, dword ptr [rip + {c1}]
    mov rax, rsi
    shl rax, 6
    lea rax, [r8 + rax*1]
    vxorps xmm2, xmm2, xmm2
    vxorps xmm3, xmm3, xmm3
    vmovaps ymmword ptr [rsp], ymm3
    vxorps xmm3, xmm3, xmm3
    vxorps xmm4, xmm4, xmm4
    vmovaps ymmword ptr [rsp + 32], ymm4
    lea r10, [rsp + 512]
    lea r11, [rsp + 768]
    # scales of superblock +0
    vmovdqu xmm4, xmmword ptr [rdi + 4]
    vpand xmm5, xmm4, xmmword ptr [rip + {c2}]
    vpsrlw xmm4, xmm4, 2
    vpand xmm4, xmm4, xmmword ptr [rip + {c3}]
    vpbroadcastd xmm6, dword ptr [rdi + 12]
    vpsrlvd xmm6, xmm6, xmmword ptr [rip + {c5}]
    vpand xmm6, xmm6, xmmword ptr [rip + {c4}]
    vpor xmm4, xmm4, xmm6
    vpunpckldq xmm5, xmm5, xmm4
    vpshufb xmm4, xmm5, xmmword ptr [rip + {c7}]
    vpshufb xmm5, xmm5, xmmword ptr [rip + {c8}]
    vpmovzxbd ymm4, xmm4
    vpmovzxbd ymm5, xmm5
    vcvtdq2ps ymm4, ymm4
    vcvtdq2ps ymm5, ymm5
    vpbroadcastd xmm6, dword ptr [rdi]
    vpxor xmm6, xmm6, xmmword ptr [rip + {c6}]
    vcvtph2ps ymm6, xmm6
    vmulps ymm4, ymm4, ymm6
    vmulps ymm5, ymm5, ymm6
    vmovaps ymmword ptr [r10], ymm4
    vmovaps ymmword ptr [r10 + 32], ymm5
    vmovsldup ymm4, ymm4
    vmulps ymm4, ymm4, ymmword ptr [rcx]
    vmovaps ymmword ptr [r10 + 64], ymm4
    vmovsldup ymm4, ymm5
    vmulps ymm4, ymm4, ymmword ptr [rcx + 32]
    vmovaps ymmword ptr [r10 + 96], ymm4
    dec rsi
    jz 3f
2:
    prefetcht1 byte ptr [rdi + 1024]
    prefetcht1 byte ptr [rdi + 1088]
    prefetcht1 byte ptr [rdi + 1152]
    # offsets
    vmovshdup ymm4, ymmword ptr [r10]
    vmovshdup ymm5, ymmword ptr [r10 + 32]
    vmovdqa ymm6, ymmword ptr [rsp]
    vfmadd231ps ymm6, ymm4, ymmword ptr [r8]
    vfmadd231ps ymm6, ymm5, ymmword ptr [r8 + 32]
    vmovaps ymmword ptr [rsp], ymm6
    vmovdqa ymm6, ymmword ptr [rsp + 32]
    vfmadd231ps ymm6, ymm4, ymmword ptr [rax]
    vfmadd231ps ymm6, ymm5, ymmword ptr [rax + 32]
    vmovaps ymmword ptr [rsp + 32], ymm6
    vmovdqu ymm4, ymmword ptr [rdi + 16]
    vpand ymm5, ymm4, ymm0
    vpsrlw ymm4, ymm4, 4
    vpand ymm4, ymm4, ymm0
    # block 0
    # scales of superblock +1
    vmovdqu xmm6, xmmword ptr [rdi + 148]
    vpand xmm7, xmm6, xmmword ptr [rip + {c2}]
    vpsrlw xmm6, xmm6, 2
    vpand xmm6, xmm6, xmmword ptr [rip + {c3}]
    vpmaddubsw ymm8, ymm5, ymmword ptr [rdx]
    vpmaddwd ymm8, ymm8, ymm1
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm9, dword ptr [r10 + 64]
    vfmadd231ps ymm2, ymm8, ymm9
    vpmaddubsw ymm5, ymm5, ymmword ptr [rdx + 32]
    vpmaddwd ymm5, ymm5, ymm1
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm8, dword ptr [r10 + 68]
    vfmadd231ps ymm3, ymm5, ymm8
    # block 1
    vpbroadcastd xmm5, dword ptr [rdi + 156]
    vpsrlvd xmm5, xmm5, xmmword ptr [rip + {c5}]
    vpand xmm5, xmm5, xmmword ptr [rip + {c4}]
    vpor xmm6, xmm6, xmm5
    vpmaddubsw ymm5, ymm4, ymmword ptr [rdx + 64]
    vpmaddwd ymm5, ymm5, ymm1
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm8, dword ptr [r10 + 72]
    vfmadd231ps ymm2, ymm5, ymm8
    vpmaddubsw ymm4, ymm4, ymmword ptr [rdx + 96]
    vpmaddwd ymm4, ymm4, ymm1
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, dword ptr [r10 + 76]
    vfmadd231ps ymm3, ymm4, ymm5
    vmovdqu ymm4, ymmword ptr [rdi + 48]
    vpand ymm5, ymm4, ymm0
    vpsrlw ymm4, ymm4, 4
    vpand ymm4, ymm4, ymm0
    # block 2
    vpunpckldq xmm7, xmm7, xmm6
    vpshufb xmm6, xmm7, xmmword ptr [rip + {c7}]
    vpshufb xmm7, xmm7, xmmword ptr [rip + {c8}]
    vpmovzxbd ymm6, xmm6
    vpmaddubsw ymm8, ymm5, ymmword ptr [rdx + 128]
    vpmaddwd ymm8, ymm8, ymm1
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm9, dword ptr [r10 + 80]
    vfmadd231ps ymm2, ymm8, ymm9
    vpmaddubsw ymm5, ymm5, ymmword ptr [rdx + 160]
    vpmaddwd ymm5, ymm5, ymm1
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm8, dword ptr [r10 + 84]
    vfmadd231ps ymm3, ymm5, ymm8
    # block 3
    vpmovzxbd ymm7, xmm7
    vcvtdq2ps ymm6, ymm6
    vcvtdq2ps ymm7, ymm7
    vpbroadcastd xmm5, dword ptr [rdi + 144]
    vpmaddubsw ymm8, ymm4, ymmword ptr [rdx + 192]
    vpmaddwd ymm8, ymm8, ymm1
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm9, dword ptr [r10 + 88]
    vfmadd231ps ymm2, ymm8, ymm9
    vpmaddubsw ymm4, ymm4, ymmword ptr [rdx + 224]
    vpmaddwd ymm4, ymm4, ymm1
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm8, dword ptr [r10 + 92]
    vfmadd231ps ymm3, ymm4, ymm8
    vmovdqu ymm4, ymmword ptr [rdi + 80]
    vpand ymm8, ymm4, ymm0
    vpsrlw ymm4, ymm4, 4
    vpand ymm4, ymm4, ymm0
    # block 4
    vpxor xmm5, xmm5, xmmword ptr [rip + {c6}]
    vcvtph2ps ymm5, xmm5
    vmulps ymm6, ymm6, ymm5
    vmulps ymm7, ymm7, ymm5
    vpmaddubsw ymm5, ymm8, ymmword ptr [rdx + 256]
    vpmaddwd ymm5, ymm5, ymm1
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm9, dword ptr [r10 + 96]
    vfmadd231ps ymm2, ymm5, ymm9
    vpmaddubsw ymm5, ymm8, ymmword ptr [rdx + 288]
    vpmaddwd ymm5, ymm5, ymm1
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm8, dword ptr [r10 + 100]
    vfmadd231ps ymm3, ymm5, ymm8
    # block 5
    vmovaps ymmword ptr [r11], ymm6
    vmovaps ymmword ptr [r11 + 32], ymm7
    vmovsldup ymm5, ymm6
    vmulps ymm5, ymm5, ymmword ptr [rcx + 64]
    vpmaddubsw ymm6, ymm4, ymmword ptr [rdx + 320]
    vpmaddwd ymm6, ymm6, ymm1
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm8, dword ptr [r10 + 104]
    vfmadd231ps ymm2, ymm6, ymm8
    vpmaddubsw ymm4, ymm4, ymmword ptr [rdx + 352]
    vpmaddwd ymm4, ymm4, ymm1
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm6, dword ptr [r10 + 108]
    vfmadd231ps ymm3, ymm4, ymm6
    vmovdqu ymm4, ymmword ptr [rdi + 112]
    vpand ymm6, ymm4, ymm0
    vpsrlw ymm4, ymm4, 4
    vpand ymm4, ymm4, ymm0
    # block 6
    vmovaps ymmword ptr [r11 + 64], ymm5
    vmovsldup ymm5, ymm7
    vmulps ymm5, ymm5, ymmword ptr [rcx + 96]
    vmovaps ymmword ptr [r11 + 96], ymm5
    vpmaddubsw ymm5, ymm6, ymmword ptr [rdx + 384]
    vpmaddwd ymm5, ymm5, ymm1
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm7, dword ptr [r10 + 112]
    vfmadd231ps ymm2, ymm5, ymm7
    vpmaddubsw ymm5, ymm6, ymmword ptr [rdx + 416]
    vpmaddwd ymm5, ymm5, ymm1
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [r10 + 116]
    vfmadd231ps ymm3, ymm5, ymm6
    # block 7
    vpmaddubsw ymm5, ymm4, ymmword ptr [rdx + 448]
    vpmaddwd ymm5, ymm5, ymm1
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [r10 + 120]
    vfmadd231ps ymm2, ymm5, ymm6
    vpmaddubsw ymm4, ymm4, ymmword ptr [rdx + 480]
    vpmaddwd ymm4, ymm4, ymm1
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, dword ptr [r10 + 124]
    vfmadd231ps ymm3, ymm4, ymm5
    add rdi, 144
    add rdx, 512
    add rcx, 64
    add r8, 64
    add rax, 64
    xor r10, 256
    xor r11, 256
    dec rsi
    jnz 2b
3:
    # offsets
    vmovshdup ymm4, ymmword ptr [r10]
    vmovshdup ymm5, ymmword ptr [r10 + 32]
    vmovdqa ymm6, ymmword ptr [rsp]
    vfmadd231ps ymm6, ymm4, ymmword ptr [r8]
    vfmadd231ps ymm6, ymm5, ymmword ptr [r8 + 32]
    vmovaps ymmword ptr [rsp], ymm6
    vmovdqa ymm6, ymmword ptr [rsp + 32]
    vfmadd231ps ymm6, ymm4, ymmword ptr [rax]
    vfmadd231ps ymm6, ymm5, ymmword ptr [rax + 32]
    vmovaps ymmword ptr [rsp + 32], ymm6
    vmovdqu ymm4, ymmword ptr [rdi + 16]
    vpand ymm5, ymm4, ymm0
    vpsrlw ymm4, ymm4, 4
    vpand ymm4, ymm4, ymm0
    # block 0
    vpmaddubsw ymm6, ymm5, ymmword ptr [rdx]
    vpmaddwd ymm6, ymm6, ymm1
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [r10 + 64]
    vfmadd231ps ymm2, ymm6, ymm7
    vpmaddubsw ymm5, ymm5, ymmword ptr [rdx + 32]
    vpmaddwd ymm5, ymm5, ymm1
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [r10 + 68]
    vfmadd231ps ymm3, ymm5, ymm6
    # block 1
    vpmaddubsw ymm5, ymm4, ymmword ptr [rdx + 64]
    vpmaddwd ymm5, ymm5, ymm1
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [r10 + 72]
    vfmadd231ps ymm2, ymm5, ymm6
    vpmaddubsw ymm4, ymm4, ymmword ptr [rdx + 96]
    vpmaddwd ymm4, ymm4, ymm1
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, dword ptr [r10 + 76]
    vfmadd231ps ymm3, ymm4, ymm5
    vmovdqu ymm4, ymmword ptr [rdi + 48]
    vpand ymm5, ymm4, ymm0
    vpsrlw ymm4, ymm4, 4
    vpand ymm4, ymm4, ymm0
    # block 2
    vpmaddubsw ymm6, ymm5, ymmword ptr [rdx + 128]
    vpmaddwd ymm6, ymm6, ymm1
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [r10 + 80]
    vfmadd231ps ymm2, ymm6, ymm7
    vpmaddubsw ymm5, ymm5, ymmword ptr [rdx + 160]
    vpmaddwd ymm5, ymm5, ymm1
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [r10 + 84]
    vfmadd231ps ymm3, ymm5, ymm6
    # block 3
    vpmaddubsw ymm5, ymm4, ymmword ptr [rdx + 192]
    vpmaddwd ymm5, ymm5, ymm1
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [r10 + 88]
    vfmadd231ps ymm2, ymm5, ymm6
    vpmaddubsw ymm4, ymm4, ymmword ptr [rdx + 224]
    vpmaddwd ymm4, ymm4, ymm1
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, dword ptr [r10 + 92]
    vfmadd231ps ymm3, ymm4, ymm5
    vmovdqu ymm4, ymmword ptr [rdi + 80]
    vpand ymm5, ymm4, ymm0
    vpsrlw ymm4, ymm4, 4
    vpand ymm4, ymm4, ymm0
    # block 4
    vpmaddubsw ymm6, ymm5, ymmword ptr [rdx + 256]
    vpmaddwd ymm6, ymm6, ymm1
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [r10 + 96]
    vfmadd231ps ymm2, ymm6, ymm7
    vpmaddubsw ymm5, ymm5, ymmword ptr [rdx + 288]
    vpmaddwd ymm5, ymm5, ymm1
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [r10 + 100]
    vfmadd231ps ymm3, ymm5, ymm6
    # block 5
    vpmaddubsw ymm5, ymm4, ymmword ptr [rdx + 320]
    vpmaddwd ymm5, ymm5, ymm1
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [r10 + 104]
    vfmadd231ps ymm2, ymm5, ymm6
    vpmaddubsw ymm4, ymm4, ymmword ptr [rdx + 352]
    vpmaddwd ymm4, ymm4, ymm1
    vcvtdq2ps ymm4, ymm4
    vbroadcastss ymm5, dword ptr [r10 + 108]
    vfmadd231ps ymm3, ymm4, ymm5
    vmovdqu ymm4, ymmword ptr [rdi + 112]
    vpand ymm5, ymm4, ymm0
    vpsrlw ymm4, ymm4, 4
    vpand ymm4, ymm4, ymm0
    # block 6
    vpmaddubsw ymm0, ymm5, ymmword ptr [rdx + 384]
    vpmaddwd ymm0, ymm0, ymm1
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm6, dword ptr [r10 + 112]
    vfmadd231ps ymm2, ymm0, ymm6
    vpmaddubsw ymm0, ymm5, ymmword ptr [rdx + 416]
    vpmaddwd ymm0, ymm0, ymm1
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm5, dword ptr [r10 + 116]
    vfmadd231ps ymm3, ymm0, ymm5
    # block 7
    vpmaddubsw ymm0, ymm4, ymmword ptr [rdx + 448]
    vpmaddwd ymm0, ymm0, ymm1
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm5, dword ptr [r10 + 120]
    vfmadd231ps ymm2, ymm0, ymm5
    vpmaddubsw ymm0, ymm4, ymmword ptr [rdx + 480]
    vpmaddwd ymm0, ymm0, ymm1
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm1, dword ptr [r10 + 124]
    vfmadd231ps ymm3, ymm0, ymm1
    vaddps ymm0, ymm2, ymmword ptr [rsp]
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vmovhlps xmm1, xmm0, xmm0
    vaddps xmm0, xmm0, xmm1
    vmovshdup xmm1, xmm0
    vaddss xmm0, xmm0, xmm1
    vmovss dword ptr [r9], xmm0
    vaddps ymm0, ymm3, ymmword ptr [rsp + 32]
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
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
# c0, aligned to 4:
#   0f 0f 0f 0f
# c1, aligned to 4:
#   01 00 01 00
# c2, aligned to 16:
#   3f 3f 3f 3f 3f 3f 3f 3f 3f 3f 3f 3f 3f 3f 3f 3f
# c3, aligned to 16:
#   30 30 30 30 30 30 30 30 30 30 30 30 30 30 30 30
# c4, aligned to 16:
#   0f 0f 0f 0f 0f 0f 0f 0f 0f 0f 0f 0f 0f 0f 0f 0f
# c5, aligned to 16:
#   00 00 00 00 04 00 00 00 00 00 00 00 04 00 00 00
# c6, aligned to 16:
#   00 00 00 80 00 00 00 80 00 00 00 80 00 00 00 80
# c7, aligned to 16:
#   00 08 01 09 02 0a 03 0b 80 80 80 80 80 80 80 80
# c8, aligned to 16:
#   04 0c 05 0d 06 0e 07 0f 80 80 80 80 80 80 80 80
