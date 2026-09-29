# q4_k_x4_ahead1024_t1(gpr, gpr, gpr, gpr, gpr, gpr) -> ()
# frame: 1024 bytes aligned to 512, saves [rbx, r12]
# row=rdi sbs=rsi quants=rdx scales=rcx half_sums=r8 out=r9
# nibble_mask=ymm0 one16=ymm1 stride=rax sums1=r10 sums2=r11 sums3=rax
# acc0=ymm2 acc1=ymm3 acc2=ymm4 acc3=ymm5 zero=ymm3,ymm4,ymm5,ymm6 cur=rbx
# nxt=r12 packed=ymm6,ymm8 low_six=ymm7,ymm9 top_bits=ymm6,ymm8 last_four=ymm8,ymm10 blocks_0_3=ymm6,ymm7
# blocks_4_7=ymm7,ymm9 d_dmin=ymm8 d_and_minus_dmin=ymm8 pair_of_blocks=ymm8,ymm6,ymm10 scale_products=ymm8,ymm6,ymm10,ymm7 offsets_lo=ymm6
# offsets_hi=ymm7 off=ymm8 codes=ymm6 low_nibbles=ymm7,ymm8 high_nibbles=ymm6 products=ymm11,ymm7,ymm8,ymm6,ymm10,ymm0
# block_scale=ymm12,ymm11,ymm10,ymm8,ymm7,ymm9,ymm1 sum=ymm0 high=ymm1 pair=ymm1 result=ymm0 nothing=ymm0
    push rbx
    push r12
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
    lea r10, [r8 + rax*1]
    lea r11, [r10 + rax*1]
    lea rax, [r11 + rax*1]
    vxorps xmm2, xmm2, xmm2
    vxorps xmm3, xmm3, xmm3
    vmovaps ymmword ptr [rsp], ymm3
    vxorps xmm3, xmm3, xmm3
    vxorps xmm4, xmm4, xmm4
    vmovaps ymmword ptr [rsp + 32], ymm4
    vxorps xmm4, xmm4, xmm4
    vxorps xmm5, xmm5, xmm5
    vmovaps ymmword ptr [rsp + 64], ymm5
    vxorps xmm5, xmm5, xmm5
    vxorps xmm6, xmm6, xmm6
    vmovaps ymmword ptr [rsp + 96], ymm6
    lea rbx, [rsp + 512]
    lea r12, [rsp + 768]
    # scales of superblock +0
    vmovdqu xmm6, xmmword ptr [rdi + 4]
    vpand xmm7, xmm6, xmmword ptr [rip + {c2}]
    vpsrlw xmm6, xmm6, 2
    vpand xmm6, xmm6, xmmword ptr [rip + {c3}]
    vpbroadcastd xmm8, dword ptr [rdi + 12]
    vpsrlvd xmm8, xmm8, xmmword ptr [rip + {c5}]
    vpand xmm8, xmm8, xmmword ptr [rip + {c4}]
    vpor xmm6, xmm6, xmm8
    vpunpckldq xmm7, xmm7, xmm6
    vpshufb xmm6, xmm7, xmmword ptr [rip + {c7}]
    vpshufb xmm7, xmm7, xmmword ptr [rip + {c8}]
    vpmovzxbd ymm6, xmm6
    vpmovzxbd ymm7, xmm7
    vcvtdq2ps ymm6, ymm6
    vcvtdq2ps ymm7, ymm7
    vpbroadcastd xmm8, dword ptr [rdi]
    vpxor xmm8, xmm8, xmmword ptr [rip + {c6}]
    vcvtph2ps ymm8, xmm8
    vmulps ymm6, ymm6, ymm8
    vmulps ymm7, ymm7, ymm8
    vmovaps ymmword ptr [rbx], ymm6
    vmovaps ymmword ptr [rbx + 32], ymm7
    vmovdqa ymm8, ymmword ptr [rip + {c9}]
    vpermps ymm8, ymm8, ymm6
    vmulps ymm8, ymm8, ymmword ptr [rcx]
    vmovaps ymmword ptr [rbx + 64], ymm8
    vmovdqa ymm8, ymmword ptr [rip + {c10}]
    vpermps ymm6, ymm8, ymm6
    vmulps ymm6, ymm6, ymmword ptr [rcx + 32]
    vmovaps ymmword ptr [rbx + 96], ymm6
    vmovdqa ymm6, ymmword ptr [rip + {c9}]
    vpermps ymm6, ymm6, ymm7
    vmulps ymm6, ymm6, ymmword ptr [rcx + 64]
    vmovaps ymmword ptr [rbx + 128], ymm6
    vmovdqa ymm6, ymmword ptr [rip + {c10}]
    vpermps ymm6, ymm6, ymm7
    vmulps ymm6, ymm6, ymmword ptr [rcx + 96]
    vmovaps ymmword ptr [rbx + 160], ymm6
    dec rsi
    jz 3f
2:
    prefetcht1 byte ptr [rdi + 1024]
    prefetcht1 byte ptr [rdi + 1088]
    prefetcht1 byte ptr [rdi + 1152]
    # offsets
    vmovshdup ymm6, ymmword ptr [rbx]
    vmovshdup ymm7, ymmword ptr [rbx + 32]
    vmovdqa ymm8, ymmword ptr [rsp]
    vfmadd231ps ymm8, ymm6, ymmword ptr [r8]
    vfmadd231ps ymm8, ymm7, ymmword ptr [r8 + 32]
    vmovaps ymmword ptr [rsp], ymm8
    vmovdqa ymm8, ymmword ptr [rsp + 32]
    vfmadd231ps ymm8, ymm6, ymmword ptr [r10]
    vfmadd231ps ymm8, ymm7, ymmword ptr [r10 + 32]
    vmovaps ymmword ptr [rsp + 32], ymm8
    vmovdqa ymm8, ymmword ptr [rsp + 64]
    vfmadd231ps ymm8, ymm6, ymmword ptr [r11]
    vfmadd231ps ymm8, ymm7, ymmword ptr [r11 + 32]
    vmovaps ymmword ptr [rsp + 64], ymm8
    vmovdqa ymm8, ymmword ptr [rsp + 96]
    vfmadd231ps ymm8, ymm6, ymmword ptr [rax]
    vfmadd231ps ymm8, ymm7, ymmword ptr [rax + 32]
    vmovaps ymmword ptr [rsp + 96], ymm8
    vmovdqu ymm6, ymmword ptr [rdi + 16]
    vpand ymm7, ymm6, ymm0
    vpsrlw ymm6, ymm6, 4
    vpand ymm6, ymm6, ymm0
    # block 0
    # scales of superblock +1
    vmovdqu xmm8, xmmword ptr [rdi + 148]
    vpand xmm9, xmm8, xmmword ptr [rip + {c2}]
    vpsrlw xmm8, xmm8, 2
    vpand xmm8, xmm8, xmmword ptr [rip + {c3}]
    vpbroadcastd xmm10, dword ptr [rdi + 156]
    vpmaddubsw ymm11, ymm7, ymmword ptr [rdx]
    vpmaddwd ymm11, ymm11, ymm1
    vcvtdq2ps ymm11, ymm11
    vbroadcastss ymm12, dword ptr [rbx + 64]
    vfmadd231ps ymm2, ymm11, ymm12
    vpmaddubsw ymm11, ymm7, ymmword ptr [rdx + 32]
    vpmaddwd ymm11, ymm11, ymm1
    vcvtdq2ps ymm11, ymm11
    vbroadcastss ymm12, dword ptr [rbx + 68]
    vfmadd231ps ymm3, ymm11, ymm12
    vpmaddubsw ymm11, ymm7, ymmword ptr [rdx + 64]
    vpmaddwd ymm11, ymm11, ymm1
    vcvtdq2ps ymm11, ymm11
    vbroadcastss ymm12, dword ptr [rbx + 72]
    vfmadd231ps ymm4, ymm11, ymm12
    vpmaddubsw ymm7, ymm7, ymmword ptr [rdx + 96]
    vpmaddwd ymm7, ymm7, ymm1
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm11, dword ptr [rbx + 76]
    vfmadd231ps ymm5, ymm7, ymm11
    # block 1
    vpsrlvd xmm10, xmm10, xmmword ptr [rip + {c5}]
    vpand xmm10, xmm10, xmmword ptr [rip + {c4}]
    vpor xmm8, xmm8, xmm10
    vpunpckldq xmm9, xmm9, xmm8
    vpshufb xmm7, xmm9, xmmword ptr [rip + {c7}]
    vpmaddubsw ymm8, ymm6, ymmword ptr [rdx + 128]
    vpmaddwd ymm8, ymm8, ymm1
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm10, dword ptr [rbx + 80]
    vfmadd231ps ymm2, ymm8, ymm10
    vpmaddubsw ymm8, ymm6, ymmword ptr [rdx + 160]
    vpmaddwd ymm8, ymm8, ymm1
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm10, dword ptr [rbx + 84]
    vfmadd231ps ymm3, ymm8, ymm10
    vpmaddubsw ymm8, ymm6, ymmword ptr [rdx + 192]
    vpmaddwd ymm8, ymm8, ymm1
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm10, dword ptr [rbx + 88]
    vfmadd231ps ymm4, ymm8, ymm10
    vpmaddubsw ymm6, ymm6, ymmword ptr [rdx + 224]
    vpmaddwd ymm6, ymm6, ymm1
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm8, dword ptr [rbx + 92]
    vfmadd231ps ymm5, ymm6, ymm8
    vmovdqu ymm6, ymmword ptr [rdi + 48]
    vpand ymm8, ymm6, ymm0
    vpsrlw ymm6, ymm6, 4
    vpand ymm6, ymm6, ymm0
    # block 2
    vpshufb xmm9, xmm9, xmmword ptr [rip + {c8}]
    vpmovzxbd ymm7, xmm7
    vpmovzxbd ymm9, xmm9
    vcvtdq2ps ymm7, ymm7
    vcvtdq2ps ymm9, ymm9
    vpmaddubsw ymm10, ymm8, ymmword ptr [rdx + 256]
    vpmaddwd ymm10, ymm10, ymm1
    vcvtdq2ps ymm10, ymm10
    vbroadcastss ymm11, dword ptr [rbx + 96]
    vfmadd231ps ymm2, ymm10, ymm11
    vpmaddubsw ymm10, ymm8, ymmword ptr [rdx + 288]
    vpmaddwd ymm10, ymm10, ymm1
    vcvtdq2ps ymm10, ymm10
    vbroadcastss ymm11, dword ptr [rbx + 100]
    vfmadd231ps ymm3, ymm10, ymm11
    vpmaddubsw ymm10, ymm8, ymmword ptr [rdx + 320]
    vpmaddwd ymm10, ymm10, ymm1
    vcvtdq2ps ymm10, ymm10
    vbroadcastss ymm11, dword ptr [rbx + 104]
    vfmadd231ps ymm4, ymm10, ymm11
    vpmaddubsw ymm8, ymm8, ymmword ptr [rdx + 352]
    vpmaddwd ymm8, ymm8, ymm1
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm10, dword ptr [rbx + 108]
    vfmadd231ps ymm5, ymm8, ymm10
    # block 3
    vpbroadcastd xmm8, dword ptr [rdi + 144]
    vpxor xmm8, xmm8, xmmword ptr [rip + {c6}]
    vcvtph2ps ymm8, xmm8
    vmulps ymm7, ymm7, ymm8
    vmulps ymm9, ymm9, ymm8
    vpmaddubsw ymm8, ymm6, ymmword ptr [rdx + 384]
    vpmaddwd ymm8, ymm8, ymm1
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm10, dword ptr [rbx + 112]
    vfmadd231ps ymm2, ymm8, ymm10
    vpmaddubsw ymm8, ymm6, ymmword ptr [rdx + 416]
    vpmaddwd ymm8, ymm8, ymm1
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm10, dword ptr [rbx + 116]
    vfmadd231ps ymm3, ymm8, ymm10
    vpmaddubsw ymm8, ymm6, ymmword ptr [rdx + 448]
    vpmaddwd ymm8, ymm8, ymm1
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm10, dword ptr [rbx + 120]
    vfmadd231ps ymm4, ymm8, ymm10
    vpmaddubsw ymm6, ymm6, ymmword ptr [rdx + 480]
    vpmaddwd ymm6, ymm6, ymm1
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm8, dword ptr [rbx + 124]
    vfmadd231ps ymm5, ymm6, ymm8
    vmovdqu ymm6, ymmword ptr [rdi + 80]
    vpand ymm8, ymm6, ymm0
    vpsrlw ymm6, ymm6, 4
    vpand ymm6, ymm6, ymm0
    # block 4
    vmovaps ymmword ptr [r12], ymm7
    vmovaps ymmword ptr [r12 + 32], ymm9
    vmovdqa ymm10, ymmword ptr [rip + {c9}]
    vpermps ymm10, ymm10, ymm7
    vmulps ymm10, ymm10, ymmword ptr [rcx + 128]
    vpmaddubsw ymm11, ymm8, ymmword ptr [rdx + 512]
    vpmaddwd ymm11, ymm11, ymm1
    vcvtdq2ps ymm11, ymm11
    vbroadcastss ymm12, dword ptr [rbx + 128]
    vfmadd231ps ymm2, ymm11, ymm12
    vpmaddubsw ymm11, ymm8, ymmword ptr [rdx + 544]
    vpmaddwd ymm11, ymm11, ymm1
    vcvtdq2ps ymm11, ymm11
    vbroadcastss ymm12, dword ptr [rbx + 132]
    vfmadd231ps ymm3, ymm11, ymm12
    vpmaddubsw ymm11, ymm8, ymmword ptr [rdx + 576]
    vpmaddwd ymm11, ymm11, ymm1
    vcvtdq2ps ymm11, ymm11
    vbroadcastss ymm12, dword ptr [rbx + 136]
    vfmadd231ps ymm4, ymm11, ymm12
    vpmaddubsw ymm8, ymm8, ymmword ptr [rdx + 608]
    vpmaddwd ymm8, ymm8, ymm1
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm11, dword ptr [rbx + 140]
    vfmadd231ps ymm5, ymm8, ymm11
    # block 5
    vmovaps ymmword ptr [r12 + 64], ymm10
    vmovdqa ymm8, ymmword ptr [rip + {c10}]
    vpermps ymm7, ymm8, ymm7
    vmulps ymm7, ymm7, ymmword ptr [rcx + 160]
    vmovaps ymmword ptr [r12 + 96], ymm7
    vpmaddubsw ymm7, ymm6, ymmword ptr [rdx + 640]
    vpmaddwd ymm7, ymm7, ymm1
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rbx + 144]
    vfmadd231ps ymm2, ymm7, ymm8
    vpmaddubsw ymm7, ymm6, ymmword ptr [rdx + 672]
    vpmaddwd ymm7, ymm7, ymm1
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rbx + 148]
    vfmadd231ps ymm3, ymm7, ymm8
    vpmaddubsw ymm7, ymm6, ymmword ptr [rdx + 704]
    vpmaddwd ymm7, ymm7, ymm1
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rbx + 152]
    vfmadd231ps ymm4, ymm7, ymm8
    vpmaddubsw ymm6, ymm6, ymmword ptr [rdx + 736]
    vpmaddwd ymm6, ymm6, ymm1
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rbx + 156]
    vfmadd231ps ymm5, ymm6, ymm7
    vmovdqu ymm6, ymmword ptr [rdi + 112]
    vpand ymm7, ymm6, ymm0
    vpsrlw ymm6, ymm6, 4
    vpand ymm6, ymm6, ymm0
    # block 6
    vmovdqa ymm8, ymmword ptr [rip + {c9}]
    vpermps ymm8, ymm8, ymm9
    vmulps ymm8, ymm8, ymmword ptr [rcx + 192]
    vmovaps ymmword ptr [r12 + 128], ymm8
    vmovdqa ymm8, ymmword ptr [rip + {c10}]
    vpmaddubsw ymm10, ymm7, ymmword ptr [rdx + 768]
    vpmaddwd ymm10, ymm10, ymm1
    vcvtdq2ps ymm10, ymm10
    vbroadcastss ymm11, dword ptr [rbx + 160]
    vfmadd231ps ymm2, ymm10, ymm11
    vpmaddubsw ymm10, ymm7, ymmword ptr [rdx + 800]
    vpmaddwd ymm10, ymm10, ymm1
    vcvtdq2ps ymm10, ymm10
    vbroadcastss ymm11, dword ptr [rbx + 164]
    vfmadd231ps ymm3, ymm10, ymm11
    vpmaddubsw ymm10, ymm7, ymmword ptr [rdx + 832]
    vpmaddwd ymm10, ymm10, ymm1
    vcvtdq2ps ymm10, ymm10
    vbroadcastss ymm11, dword ptr [rbx + 168]
    vfmadd231ps ymm4, ymm10, ymm11
    vpmaddubsw ymm7, ymm7, ymmword ptr [rdx + 864]
    vpmaddwd ymm7, ymm7, ymm1
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm10, dword ptr [rbx + 172]
    vfmadd231ps ymm5, ymm7, ymm10
    # block 7
    vpermps ymm7, ymm8, ymm9
    vmulps ymm7, ymm7, ymmword ptr [rcx + 224]
    vmovaps ymmword ptr [r12 + 160], ymm7
    vpmaddubsw ymm7, ymm6, ymmword ptr [rdx + 896]
    vpmaddwd ymm7, ymm7, ymm1
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rbx + 176]
    vfmadd231ps ymm2, ymm7, ymm8
    vpmaddubsw ymm7, ymm6, ymmword ptr [rdx + 928]
    vpmaddwd ymm7, ymm7, ymm1
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rbx + 180]
    vfmadd231ps ymm3, ymm7, ymm8
    vpmaddubsw ymm7, ymm6, ymmword ptr [rdx + 960]
    vpmaddwd ymm7, ymm7, ymm1
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rbx + 184]
    vfmadd231ps ymm4, ymm7, ymm8
    vpmaddubsw ymm6, ymm6, ymmword ptr [rdx + 992]
    vpmaddwd ymm6, ymm6, ymm1
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rbx + 188]
    vfmadd231ps ymm5, ymm6, ymm7
    add rdi, 144
    add rdx, 1024
    add rcx, 128
    add r8, 64
    add r10, 64
    add r11, 64
    add rax, 64
    xor rbx, 256
    xor r12, 256
    dec rsi
    jnz 2b
3:
    # offsets
    vmovshdup ymm6, ymmword ptr [rbx]
    vmovshdup ymm7, ymmword ptr [rbx + 32]
    vmovdqa ymm8, ymmword ptr [rsp]
    vfmadd231ps ymm8, ymm6, ymmword ptr [r8]
    vfmadd231ps ymm8, ymm7, ymmword ptr [r8 + 32]
    vmovaps ymmword ptr [rsp], ymm8
    vmovdqa ymm8, ymmword ptr [rsp + 32]
    vfmadd231ps ymm8, ymm6, ymmword ptr [r10]
    vfmadd231ps ymm8, ymm7, ymmword ptr [r10 + 32]
    vmovaps ymmword ptr [rsp + 32], ymm8
    vmovdqa ymm8, ymmword ptr [rsp + 64]
    vfmadd231ps ymm8, ymm6, ymmword ptr [r11]
    vfmadd231ps ymm8, ymm7, ymmword ptr [r11 + 32]
    vmovaps ymmword ptr [rsp + 64], ymm8
    vmovdqa ymm8, ymmword ptr [rsp + 96]
    vfmadd231ps ymm8, ymm6, ymmword ptr [rax]
    vfmadd231ps ymm8, ymm7, ymmword ptr [rax + 32]
    vmovaps ymmword ptr [rsp + 96], ymm8
    vmovdqu ymm6, ymmword ptr [rdi + 16]
    vpand ymm7, ymm6, ymm0
    vpsrlw ymm6, ymm6, 4
    vpand ymm6, ymm6, ymm0
    # block 0
    vpmaddubsw ymm8, ymm7, ymmword ptr [rdx]
    vpmaddwd ymm8, ymm8, ymm1
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm9, dword ptr [rbx + 64]
    vfmadd231ps ymm2, ymm8, ymm9
    vpmaddubsw ymm8, ymm7, ymmword ptr [rdx + 32]
    vpmaddwd ymm8, ymm8, ymm1
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm9, dword ptr [rbx + 68]
    vfmadd231ps ymm3, ymm8, ymm9
    vpmaddubsw ymm8, ymm7, ymmword ptr [rdx + 64]
    vpmaddwd ymm8, ymm8, ymm1
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm9, dword ptr [rbx + 72]
    vfmadd231ps ymm4, ymm8, ymm9
    vpmaddubsw ymm7, ymm7, ymmword ptr [rdx + 96]
    vpmaddwd ymm7, ymm7, ymm1
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rbx + 76]
    vfmadd231ps ymm5, ymm7, ymm8
    # block 1
    vpmaddubsw ymm7, ymm6, ymmword ptr [rdx + 128]
    vpmaddwd ymm7, ymm7, ymm1
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rbx + 80]
    vfmadd231ps ymm2, ymm7, ymm8
    vpmaddubsw ymm7, ymm6, ymmword ptr [rdx + 160]
    vpmaddwd ymm7, ymm7, ymm1
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rbx + 84]
    vfmadd231ps ymm3, ymm7, ymm8
    vpmaddubsw ymm7, ymm6, ymmword ptr [rdx + 192]
    vpmaddwd ymm7, ymm7, ymm1
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rbx + 88]
    vfmadd231ps ymm4, ymm7, ymm8
    vpmaddubsw ymm6, ymm6, ymmword ptr [rdx + 224]
    vpmaddwd ymm6, ymm6, ymm1
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rbx + 92]
    vfmadd231ps ymm5, ymm6, ymm7
    vmovdqu ymm6, ymmword ptr [rdi + 48]
    vpand ymm7, ymm6, ymm0
    vpsrlw ymm6, ymm6, 4
    vpand ymm6, ymm6, ymm0
    # block 2
    vpmaddubsw ymm8, ymm7, ymmword ptr [rdx + 256]
    vpmaddwd ymm8, ymm8, ymm1
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm9, dword ptr [rbx + 96]
    vfmadd231ps ymm2, ymm8, ymm9
    vpmaddubsw ymm8, ymm7, ymmword ptr [rdx + 288]
    vpmaddwd ymm8, ymm8, ymm1
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm9, dword ptr [rbx + 100]
    vfmadd231ps ymm3, ymm8, ymm9
    vpmaddubsw ymm8, ymm7, ymmword ptr [rdx + 320]
    vpmaddwd ymm8, ymm8, ymm1
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm9, dword ptr [rbx + 104]
    vfmadd231ps ymm4, ymm8, ymm9
    vpmaddubsw ymm7, ymm7, ymmword ptr [rdx + 352]
    vpmaddwd ymm7, ymm7, ymm1
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rbx + 108]
    vfmadd231ps ymm5, ymm7, ymm8
    # block 3
    vpmaddubsw ymm7, ymm6, ymmword ptr [rdx + 384]
    vpmaddwd ymm7, ymm7, ymm1
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rbx + 112]
    vfmadd231ps ymm2, ymm7, ymm8
    vpmaddubsw ymm7, ymm6, ymmword ptr [rdx + 416]
    vpmaddwd ymm7, ymm7, ymm1
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rbx + 116]
    vfmadd231ps ymm3, ymm7, ymm8
    vpmaddubsw ymm7, ymm6, ymmword ptr [rdx + 448]
    vpmaddwd ymm7, ymm7, ymm1
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rbx + 120]
    vfmadd231ps ymm4, ymm7, ymm8
    vpmaddubsw ymm6, ymm6, ymmword ptr [rdx + 480]
    vpmaddwd ymm6, ymm6, ymm1
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rbx + 124]
    vfmadd231ps ymm5, ymm6, ymm7
    vmovdqu ymm6, ymmword ptr [rdi + 80]
    vpand ymm7, ymm6, ymm0
    vpsrlw ymm6, ymm6, 4
    vpand ymm6, ymm6, ymm0
    # block 4
    vpmaddubsw ymm8, ymm7, ymmword ptr [rdx + 512]
    vpmaddwd ymm8, ymm8, ymm1
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm9, dword ptr [rbx + 128]
    vfmadd231ps ymm2, ymm8, ymm9
    vpmaddubsw ymm8, ymm7, ymmword ptr [rdx + 544]
    vpmaddwd ymm8, ymm8, ymm1
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm9, dword ptr [rbx + 132]
    vfmadd231ps ymm3, ymm8, ymm9
    vpmaddubsw ymm8, ymm7, ymmword ptr [rdx + 576]
    vpmaddwd ymm8, ymm8, ymm1
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm9, dword ptr [rbx + 136]
    vfmadd231ps ymm4, ymm8, ymm9
    vpmaddubsw ymm7, ymm7, ymmword ptr [rdx + 608]
    vpmaddwd ymm7, ymm7, ymm1
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rbx + 140]
    vfmadd231ps ymm5, ymm7, ymm8
    # block 5
    vpmaddubsw ymm7, ymm6, ymmword ptr [rdx + 640]
    vpmaddwd ymm7, ymm7, ymm1
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rbx + 144]
    vfmadd231ps ymm2, ymm7, ymm8
    vpmaddubsw ymm7, ymm6, ymmword ptr [rdx + 672]
    vpmaddwd ymm7, ymm7, ymm1
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rbx + 148]
    vfmadd231ps ymm3, ymm7, ymm8
    vpmaddubsw ymm7, ymm6, ymmword ptr [rdx + 704]
    vpmaddwd ymm7, ymm7, ymm1
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rbx + 152]
    vfmadd231ps ymm4, ymm7, ymm8
    vpmaddubsw ymm6, ymm6, ymmword ptr [rdx + 736]
    vpmaddwd ymm6, ymm6, ymm1
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rbx + 156]
    vfmadd231ps ymm5, ymm6, ymm7
    vmovdqu ymm6, ymmword ptr [rdi + 112]
    vpand ymm7, ymm6, ymm0
    vpsrlw ymm6, ymm6, 4
    vpand ymm6, ymm6, ymm0
    # block 6
    vpmaddubsw ymm0, ymm7, ymmword ptr [rdx + 768]
    vpmaddwd ymm0, ymm0, ymm1
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm8, dword ptr [rbx + 160]
    vfmadd231ps ymm2, ymm0, ymm8
    vpmaddubsw ymm0, ymm7, ymmword ptr [rdx + 800]
    vpmaddwd ymm0, ymm0, ymm1
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm8, dword ptr [rbx + 164]
    vfmadd231ps ymm3, ymm0, ymm8
    vpmaddubsw ymm0, ymm7, ymmword ptr [rdx + 832]
    vpmaddwd ymm0, ymm0, ymm1
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm8, dword ptr [rbx + 168]
    vfmadd231ps ymm4, ymm0, ymm8
    vpmaddubsw ymm0, ymm7, ymmword ptr [rdx + 864]
    vpmaddwd ymm0, ymm0, ymm1
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm7, dword ptr [rbx + 172]
    vfmadd231ps ymm5, ymm0, ymm7
    # block 7
    vpmaddubsw ymm0, ymm6, ymmword ptr [rdx + 896]
    vpmaddwd ymm0, ymm0, ymm1
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm7, dword ptr [rbx + 176]
    vfmadd231ps ymm2, ymm0, ymm7
    vpmaddubsw ymm0, ymm6, ymmword ptr [rdx + 928]
    vpmaddwd ymm0, ymm0, ymm1
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm7, dword ptr [rbx + 180]
    vfmadd231ps ymm3, ymm0, ymm7
    vpmaddubsw ymm0, ymm6, ymmword ptr [rdx + 960]
    vpmaddwd ymm0, ymm0, ymm1
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm7, dword ptr [rbx + 184]
    vfmadd231ps ymm4, ymm0, ymm7
    vpmaddubsw ymm0, ymm6, ymmword ptr [rdx + 992]
    vpmaddwd ymm0, ymm0, ymm1
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm1, dword ptr [rbx + 188]
    vfmadd231ps ymm5, ymm0, ymm1
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
    vaddps ymm0, ymm4, ymmword ptr [rsp + 64]
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vmovhlps xmm1, xmm0, xmm0
    vaddps xmm0, xmm0, xmm1
    vmovshdup xmm1, xmm0
    vaddss xmm0, xmm0, xmm1
    vmovss dword ptr [r9 + 8], xmm0
    vaddps ymm0, ymm5, ymmword ptr [rsp + 96]
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vmovhlps xmm1, xmm0, xmm0
    vaddps xmm0, xmm0, xmm1
    vmovshdup xmm1, xmm0
    vaddss xmm0, xmm0, xmm1
    vmovss dword ptr [r9 + 12], xmm0
    mov rsp, rbp
    pop rbp
    pop r12
    pop rbx
    vzeroupper
    ret
4:
    vxorps xmm0, xmm0, xmm0
    vmovss dword ptr [r9], xmm0
    vmovss dword ptr [r9 + 4], xmm0
    vmovss dword ptr [r9 + 8], xmm0
    vmovss dword ptr [r9 + 12], xmm0
    mov rsp, rbp
    pop rbp
    pop r12
    pop rbx
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
# c9, aligned to 32:
#   00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
#   02 00 00 00 02 00 00 00 02 00 00 00 02 00 00 00
# c10, aligned to 32:
#   04 00 00 00 04 00 00 00 04 00 00 00 04 00 00 00
#   06 00 00 00 06 00 00 00 06 00 00 00 06 00 00 00
