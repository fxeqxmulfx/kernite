# q6_k_front_x2(gpr, gpr, gpr, gpr, gpr, gpr) -> ()
# frame: 4096 bytes aligned to 2048, saves []
# row=rdi sbs=rsi quants=rdx scales=rcx half_sums=r8 out=r9
# stride=rax sums1=rax acc0=ymm0 acc1=ymm1 zero=ymm1,ymm2 cur=r10
# nxt=r11 one16=ymm2 lane=ymm3 half_d=ymm4,ymm8 d=ymm4,ymm8 d_times_scales=ymm5,ymm9
# spread=ymm4,ymm7,ymm5 scale_products=ymm4,ymm7,ymm5 activation_scales=ymm5,ymm8,ymm6,ymm9 offsets_lo=ymm4 offsets_hi=ymm5 off=ymm6
# nibbles_even=ymm4 nibbles_odd=ymm5,ymm6 bit_pairs=ymm6,ymm7 weights=ymm7,ymm4,ymm8 bit_pair=ymm8,ymm10,ymm7,ymm5,ymm9 products=ymm10,ymm7,ymm8,ymm4,ymm6,ymm9,ymm5
# block_scale=ymm11,ymm10,ymm9,ymm8,ymm7,ymm6,ymm5,ymm2 sum=ymm0 high=ymm2,ymm1 pair=ymm2,ymm1 result=ymm0 nothing=ymm0
    push rbp
    mov rbp, rsp
    and rsp, -2048
    sub rsp, 4096
    test rsi, rsi
    jz 4f
    # constants
    mov rax, rsi
    shl rax, 6
    lea rax, [r8 + rax*1]
    vxorps xmm0, xmm0, xmm0
    vxorps xmm1, xmm1, xmm1
    vmovaps ymmword ptr [rsp], ymm1
    vxorps xmm1, xmm1, xmm1
    vxorps xmm2, xmm2, xmm2
    vmovaps ymmword ptr [rsp + 32], ymm2
    lea r10, [rsp + 2048]
    lea r11, [rsp + 3072]
    vmovdqa ymm2, ymmword ptr [rip + {c7}]
    vmovdqa ymm3, ymmword ptr [rip + {c4}]
    # scales of superblock +0
    vpbroadcastw xmm4, word ptr [rdi]
    vcvtph2ps ymm4, xmm4
    vpmovsxbd ymm5, qword ptr [rdi + 2]
    vcvtdq2ps ymm5, ymm5
    vmulps ymm5, ymm5, ymm4
    vmovaps ymmword ptr [r10 + 512], ymm5
    vmulps ymm5, ymm5, ymmword ptr [rip + {c8}]
    vmovaps ymmword ptr [r10 + 576], ymm5
    vpmovsxbd ymm5, qword ptr [rdi + 10]
    vcvtdq2ps ymm5, ymm5
    vmulps ymm5, ymm5, ymm4
    vmovaps ymmword ptr [r10 + 544], ymm5
    vmulps ymm5, ymm5, ymmword ptr [rip + {c8}]
    vmovaps ymmword ptr [r10 + 608], ymm5
    vmovdqa ymm4, ymmword ptr [rip + {c0}]
    vpermps ymm4, ymm4, ymmword ptr [r10 + 512]
    vbroadcastsd ymm5, qword ptr [rcx]
    vmulps ymm4, ymm4, ymm5
    vmovaps ymmword ptr [r10], ymm4
    vmovdqa ymm4, ymmword ptr [rip + {c1}]
    vpermps ymm4, ymm4, ymmword ptr [r10 + 512]
    vbroadcastsd ymm5, qword ptr [rcx + 8]
    vmulps ymm4, ymm4, ymm5
    vmovaps ymmword ptr [r10 + 64], ymm4
    vmovdqa ymm4, ymmword ptr [rip + {c2}]
    vpermps ymm4, ymm4, ymmword ptr [r10 + 512]
    vbroadcastsd ymm5, qword ptr [rcx + 16]
    vmulps ymm4, ymm4, ymm5
    vmovaps ymmword ptr [r10 + 128], ymm4
    vmovdqa ymm4, ymmword ptr [rip + {c3}]
    vpermps ymm4, ymm4, ymmword ptr [r10 + 512]
    vbroadcastsd ymm5, qword ptr [rcx + 24]
    vmulps ymm4, ymm4, ymm5
    vmovaps ymmword ptr [r10 + 192], ymm4
    vmovdqa ymm4, ymmword ptr [rip + {c0}]
    vpermps ymm4, ymm4, ymmword ptr [r10 + 544]
    vbroadcastsd ymm5, qword ptr [rcx + 32]
    vmulps ymm4, ymm4, ymm5
    vmovaps ymmword ptr [r10 + 256], ymm4
    vmovdqa ymm4, ymmword ptr [rip + {c1}]
    vpermps ymm4, ymm4, ymmword ptr [r10 + 544]
    vbroadcastsd ymm5, qword ptr [rcx + 40]
    vmulps ymm4, ymm4, ymm5
    vmovaps ymmword ptr [r10 + 320], ymm4
    vmovdqa ymm4, ymmword ptr [rip + {c2}]
    vpermps ymm4, ymm4, ymmword ptr [r10 + 544]
    vbroadcastsd ymm5, qword ptr [rcx + 48]
    vmulps ymm4, ymm4, ymm5
    vmovaps ymmword ptr [r10 + 384], ymm4
    vmovdqa ymm4, ymmword ptr [rip + {c3}]
    vpermps ymm4, ymm4, ymmword ptr [r10 + 544]
    vbroadcastsd ymm5, qword ptr [rcx + 56]
    vmulps ymm4, ymm4, ymm5
    vmovaps ymmword ptr [r10 + 448], ymm4
    dec rsi
    jz 3f
2:
    # offsets
    vmovdqa ymm4, ymmword ptr [r10 + 576]
    vmovdqa ymm5, ymmword ptr [r10 + 608]
    vmovdqa ymm6, ymmword ptr [rsp]
    vfmadd231ps ymm6, ymm4, ymmword ptr [r8]
    vfmadd231ps ymm6, ymm5, ymmword ptr [r8 + 32]
    vmovaps ymmword ptr [rsp], ymm6
    vmovdqa ymm6, ymmword ptr [rsp + 32]
    vfmadd231ps ymm6, ymm4, ymmword ptr [rax]
    vfmadd231ps ymm6, ymm5, ymmword ptr [rax + 32]
    vmovaps ymmword ptr [rsp + 32], ymm6
    vmovdqu ymm4, ymmword ptr [rdi + 32]
    vmovdqu ymm5, ymmword ptr [rdi + 64]
    vmovdqu ymm6, ymmword ptr [rdi + 160]
    # block 0
    vpand ymm7, ymm4, ymmword ptr [rip + {c5}]
    vpsllw ymm8, ymm6, 4
    vpand ymm8, ymm8, ymmword ptr [rip + {c6}]
    vpor ymm7, ymm7, ymm8
    # scales of superblock +1
    vpbroadcastw xmm8, word ptr [rdi + 224]
    vcvtph2ps ymm8, xmm8
    vpmovsxbd ymm9, qword ptr [rdi + 226]
    vcvtdq2ps ymm9, ymm9
    vmulps ymm9, ymm9, ymm8
    vmovaps ymmword ptr [r11 + 512], ymm9
    vmulps ymm9, ymm9, ymmword ptr [rip + {c8}]
    vpmaddubsw ymm10, ymm7, ymmword ptr [rdx]
    vpmaddwd ymm10, ymm10, ymm2
    vcvtdq2ps ymm10, ymm10
    vpermps ymm11, ymm3, ymmword ptr [r10]
    vfmadd231ps ymm0, ymm10, ymm11
    vpmaddubsw ymm7, ymm7, ymmword ptr [rdx + 32]
    vpmaddwd ymm7, ymm7, ymm2
    vcvtdq2ps ymm7, ymm7
    vpermps ymm10, ymm3, ymmword ptr [r10 + 4]
    vfmadd231ps ymm1, ymm7, ymm10
    # block 1
    vpand ymm7, ymm5, ymmword ptr [rip + {c5}]
    vpsllw ymm10, ymm6, 2
    vpand ymm10, ymm10, ymmword ptr [rip + {c6}]
    vpor ymm7, ymm7, ymm10
    vmovaps ymmword ptr [r11 + 576], ymm9
    vpmovsxbd ymm9, qword ptr [rdi + 234]
    vcvtdq2ps ymm9, ymm9
    vmulps ymm9, ymm9, ymm8
    vmovaps ymmword ptr [r11 + 544], ymm9
    vmulps ymm9, ymm9, ymmword ptr [rip + {c8}]
    vmovaps ymmword ptr [r11 + 608], ymm9
    vpmaddubsw ymm8, ymm7, ymmword ptr [rdx + 64]
    vpmaddwd ymm8, ymm8, ymm2
    vcvtdq2ps ymm8, ymm8
    vpermps ymm9, ymm3, ymmword ptr [r10 + 64]
    vfmadd231ps ymm0, ymm8, ymm9
    vpmaddubsw ymm7, ymm7, ymmword ptr [rdx + 96]
    vpmaddwd ymm7, ymm7, ymm2
    vcvtdq2ps ymm7, ymm7
    vpermps ymm8, ymm3, ymmword ptr [r10 + 68]
    vfmadd231ps ymm1, ymm7, ymm8
    # block 2
    vpsrlw ymm4, ymm4, 4
    vpand ymm4, ymm4, ymmword ptr [rip + {c5}]
    vpand ymm7, ymm6, ymmword ptr [rip + {c6}]
    vpor ymm4, ymm4, ymm7
    vmovdqa ymm7, ymmword ptr [rip + {c0}]
    vpermps ymm7, ymm7, ymmword ptr [r11 + 512]
    vbroadcastsd ymm8, qword ptr [rcx + 64]
    vmulps ymm7, ymm7, ymm8
    vmovaps ymmword ptr [r11], ymm7
    vmovdqa ymm7, ymmword ptr [rip + {c1}]
    vpermps ymm7, ymm7, ymmword ptr [r11 + 512]
    vpmaddubsw ymm8, ymm4, ymmword ptr [rdx + 128]
    vpmaddwd ymm8, ymm8, ymm2
    vcvtdq2ps ymm8, ymm8
    vpermps ymm9, ymm3, ymmword ptr [r10 + 128]
    vfmadd231ps ymm0, ymm8, ymm9
    vpmaddubsw ymm4, ymm4, ymmword ptr [rdx + 160]
    vpmaddwd ymm4, ymm4, ymm2
    vcvtdq2ps ymm4, ymm4
    vpermps ymm8, ymm3, ymmword ptr [r10 + 132]
    vfmadd231ps ymm1, ymm4, ymm8
    # block 3
    vpsrlw ymm4, ymm5, 4
    vpand ymm4, ymm4, ymmword ptr [rip + {c5}]
    vpsrlw ymm5, ymm6, 2
    vpand ymm5, ymm5, ymmword ptr [rip + {c6}]
    vpor ymm4, ymm4, ymm5
    vbroadcastsd ymm5, qword ptr [rcx + 72]
    vmulps ymm7, ymm7, ymm5
    vmovaps ymmword ptr [r11 + 64], ymm7
    vmovdqa ymm5, ymmword ptr [rip + {c2}]
    vpermps ymm5, ymm5, ymmword ptr [r11 + 512]
    vbroadcastsd ymm6, qword ptr [rcx + 80]
    vmulps ymm5, ymm5, ymm6
    vpmaddubsw ymm6, ymm4, ymmword ptr [rdx + 192]
    vpmaddwd ymm6, ymm6, ymm2
    vcvtdq2ps ymm6, ymm6
    vpermps ymm7, ymm3, ymmword ptr [r10 + 192]
    vfmadd231ps ymm0, ymm6, ymm7
    vpmaddubsw ymm4, ymm4, ymmword ptr [rdx + 224]
    vpmaddwd ymm4, ymm4, ymm2
    vcvtdq2ps ymm4, ymm4
    vpermps ymm6, ymm3, ymmword ptr [r10 + 196]
    vfmadd231ps ymm1, ymm4, ymm6
    vmovdqu ymm4, ymmword ptr [rdi + 96]
    vmovdqu ymm6, ymmword ptr [rdi + 128]
    vmovdqu ymm7, ymmword ptr [rdi + 192]
    # block 4
    vpand ymm8, ymm4, ymmword ptr [rip + {c5}]
    vpsllw ymm9, ymm7, 4
    vpand ymm9, ymm9, ymmword ptr [rip + {c6}]
    vpor ymm8, ymm8, ymm9
    vmovaps ymmword ptr [r11 + 128], ymm5
    vmovdqa ymm5, ymmword ptr [rip + {c3}]
    vpermps ymm5, ymm5, ymmword ptr [r11 + 512]
    vbroadcastsd ymm9, qword ptr [rcx + 88]
    vmulps ymm5, ymm5, ymm9
    vmovaps ymmword ptr [r11 + 192], ymm5
    vmovdqa ymm5, ymmword ptr [rip + {c0}]
    vpmaddubsw ymm9, ymm8, ymmword ptr [rdx + 256]
    vpmaddwd ymm9, ymm9, ymm2
    vcvtdq2ps ymm9, ymm9
    vpermps ymm10, ymm3, ymmword ptr [r10 + 256]
    vfmadd231ps ymm0, ymm9, ymm10
    vpmaddubsw ymm8, ymm8, ymmword ptr [rdx + 288]
    vpmaddwd ymm8, ymm8, ymm2
    vcvtdq2ps ymm8, ymm8
    vpermps ymm9, ymm3, ymmword ptr [r10 + 260]
    vfmadd231ps ymm1, ymm8, ymm9
    # block 5
    vpand ymm8, ymm6, ymmword ptr [rip + {c5}]
    vpsllw ymm9, ymm7, 2
    vpand ymm9, ymm9, ymmword ptr [rip + {c6}]
    vpor ymm8, ymm8, ymm9
    vpermps ymm5, ymm5, ymmword ptr [r11 + 544]
    vbroadcastsd ymm9, qword ptr [rcx + 96]
    vmulps ymm5, ymm5, ymm9
    vmovaps ymmword ptr [r11 + 256], ymm5
    vmovdqa ymm5, ymmword ptr [rip + {c1}]
    vpermps ymm5, ymm5, ymmword ptr [r11 + 544]
    vbroadcastsd ymm9, qword ptr [rcx + 104]
    vpmaddubsw ymm10, ymm8, ymmword ptr [rdx + 320]
    vpmaddwd ymm10, ymm10, ymm2
    vcvtdq2ps ymm10, ymm10
    vpermps ymm11, ymm3, ymmword ptr [r10 + 320]
    vfmadd231ps ymm0, ymm10, ymm11
    vpmaddubsw ymm8, ymm8, ymmword ptr [rdx + 352]
    vpmaddwd ymm8, ymm8, ymm2
    vcvtdq2ps ymm8, ymm8
    vpermps ymm10, ymm3, ymmword ptr [r10 + 324]
    vfmadd231ps ymm1, ymm8, ymm10
    # block 6
    vpsrlw ymm4, ymm4, 4
    vpand ymm4, ymm4, ymmword ptr [rip + {c5}]
    vpand ymm8, ymm7, ymmword ptr [rip + {c6}]
    vpor ymm4, ymm4, ymm8
    vmulps ymm5, ymm5, ymm9
    vmovaps ymmword ptr [r11 + 320], ymm5
    vmovdqa ymm5, ymmword ptr [rip + {c2}]
    vpermps ymm5, ymm5, ymmword ptr [r11 + 544]
    vbroadcastsd ymm8, qword ptr [rcx + 112]
    vmulps ymm5, ymm5, ymm8
    vmovaps ymmword ptr [r11 + 384], ymm5
    vpmaddubsw ymm5, ymm4, ymmword ptr [rdx + 384]
    vpmaddwd ymm5, ymm5, ymm2
    vcvtdq2ps ymm5, ymm5
    vpermps ymm8, ymm3, ymmword ptr [r10 + 384]
    vfmadd231ps ymm0, ymm5, ymm8
    vpmaddubsw ymm4, ymm4, ymmword ptr [rdx + 416]
    vpmaddwd ymm4, ymm4, ymm2
    vcvtdq2ps ymm4, ymm4
    vpermps ymm5, ymm3, ymmword ptr [r10 + 388]
    vfmadd231ps ymm1, ymm4, ymm5
    # block 7
    vpsrlw ymm4, ymm6, 4
    vpand ymm4, ymm4, ymmword ptr [rip + {c5}]
    vpsrlw ymm5, ymm7, 2
    vpand ymm5, ymm5, ymmword ptr [rip + {c6}]
    vpor ymm4, ymm4, ymm5
    vmovdqa ymm5, ymmword ptr [rip + {c3}]
    vpermps ymm5, ymm5, ymmword ptr [r11 + 544]
    vbroadcastsd ymm6, qword ptr [rcx + 120]
    vmulps ymm5, ymm5, ymm6
    vmovaps ymmword ptr [r11 + 448], ymm5
    vpmaddubsw ymm5, ymm4, ymmword ptr [rdx + 448]
    vpmaddwd ymm5, ymm5, ymm2
    vcvtdq2ps ymm5, ymm5
    vpermps ymm6, ymm3, ymmword ptr [r10 + 448]
    vfmadd231ps ymm0, ymm5, ymm6
    vpmaddubsw ymm4, ymm4, ymmword ptr [rdx + 480]
    vpmaddwd ymm4, ymm4, ymm2
    vcvtdq2ps ymm4, ymm4
    vpermps ymm5, ymm3, ymmword ptr [r10 + 452]
    vfmadd231ps ymm1, ymm4, ymm5
    add rdi, 224
    add rdx, 512
    add rcx, 64
    add r8, 64
    add rax, 64
    xor r10, 1024
    xor r11, 1024
    dec rsi
    jnz 2b
3:
    # offsets
    vmovdqa ymm4, ymmword ptr [r10 + 576]
    vmovdqa ymm5, ymmword ptr [r10 + 608]
    vmovdqa ymm6, ymmword ptr [rsp]
    vfmadd231ps ymm6, ymm4, ymmword ptr [r8]
    vfmadd231ps ymm6, ymm5, ymmword ptr [r8 + 32]
    vmovaps ymmword ptr [rsp], ymm6
    vmovdqa ymm6, ymmword ptr [rsp + 32]
    vfmadd231ps ymm6, ymm4, ymmword ptr [rax]
    vfmadd231ps ymm6, ymm5, ymmword ptr [rax + 32]
    vmovaps ymmword ptr [rsp + 32], ymm6
    vmovdqu ymm4, ymmword ptr [rdi + 32]
    vmovdqu ymm5, ymmword ptr [rdi + 64]
    vmovdqu ymm6, ymmword ptr [rdi + 160]
    # block 0
    vpand ymm7, ymm4, ymmword ptr [rip + {c5}]
    vpsllw ymm8, ymm6, 4
    vpand ymm8, ymm8, ymmword ptr [rip + {c6}]
    vpor ymm7, ymm7, ymm8
    vpmaddubsw ymm8, ymm7, ymmword ptr [rdx]
    vpmaddwd ymm8, ymm8, ymm2
    vcvtdq2ps ymm8, ymm8
    vpermps ymm9, ymm3, ymmword ptr [r10]
    vfmadd231ps ymm0, ymm8, ymm9
    vpmaddubsw ymm7, ymm7, ymmword ptr [rdx + 32]
    vpmaddwd ymm7, ymm7, ymm2
    vcvtdq2ps ymm7, ymm7
    vpermps ymm8, ymm3, ymmword ptr [r10 + 4]
    vfmadd231ps ymm1, ymm7, ymm8
    # block 1
    vpand ymm7, ymm5, ymmword ptr [rip + {c5}]
    vpsllw ymm8, ymm6, 2
    vpand ymm8, ymm8, ymmword ptr [rip + {c6}]
    vpor ymm7, ymm7, ymm8
    vpmaddubsw ymm8, ymm7, ymmword ptr [rdx + 64]
    vpmaddwd ymm8, ymm8, ymm2
    vcvtdq2ps ymm8, ymm8
    vpermps ymm9, ymm3, ymmword ptr [r10 + 64]
    vfmadd231ps ymm0, ymm8, ymm9
    vpmaddubsw ymm7, ymm7, ymmword ptr [rdx + 96]
    vpmaddwd ymm7, ymm7, ymm2
    vcvtdq2ps ymm7, ymm7
    vpermps ymm8, ymm3, ymmword ptr [r10 + 68]
    vfmadd231ps ymm1, ymm7, ymm8
    # block 2
    vpsrlw ymm4, ymm4, 4
    vpand ymm4, ymm4, ymmword ptr [rip + {c5}]
    vpand ymm7, ymm6, ymmword ptr [rip + {c6}]
    vpor ymm4, ymm4, ymm7
    vpmaddubsw ymm7, ymm4, ymmword ptr [rdx + 128]
    vpmaddwd ymm7, ymm7, ymm2
    vcvtdq2ps ymm7, ymm7
    vpermps ymm8, ymm3, ymmword ptr [r10 + 128]
    vfmadd231ps ymm0, ymm7, ymm8
    vpmaddubsw ymm4, ymm4, ymmword ptr [rdx + 160]
    vpmaddwd ymm4, ymm4, ymm2
    vcvtdq2ps ymm4, ymm4
    vpermps ymm7, ymm3, ymmword ptr [r10 + 132]
    vfmadd231ps ymm1, ymm4, ymm7
    # block 3
    vpsrlw ymm4, ymm5, 4
    vpand ymm4, ymm4, ymmword ptr [rip + {c5}]
    vpsrlw ymm5, ymm6, 2
    vpand ymm5, ymm5, ymmword ptr [rip + {c6}]
    vpor ymm4, ymm4, ymm5
    vpmaddubsw ymm5, ymm4, ymmword ptr [rdx + 192]
    vpmaddwd ymm5, ymm5, ymm2
    vcvtdq2ps ymm5, ymm5
    vpermps ymm6, ymm3, ymmword ptr [r10 + 192]
    vfmadd231ps ymm0, ymm5, ymm6
    vpmaddubsw ymm4, ymm4, ymmword ptr [rdx + 224]
    vpmaddwd ymm4, ymm4, ymm2
    vcvtdq2ps ymm4, ymm4
    vpermps ymm5, ymm3, ymmword ptr [r10 + 196]
    vfmadd231ps ymm1, ymm4, ymm5
    vmovdqu ymm4, ymmword ptr [rdi + 96]
    vmovdqu ymm5, ymmword ptr [rdi + 128]
    vmovdqu ymm6, ymmword ptr [rdi + 192]
    # block 4
    vpand ymm7, ymm4, ymmword ptr [rip + {c5}]
    vpsllw ymm8, ymm6, 4
    vpand ymm8, ymm8, ymmword ptr [rip + {c6}]
    vpor ymm7, ymm7, ymm8
    vpmaddubsw ymm8, ymm7, ymmword ptr [rdx + 256]
    vpmaddwd ymm8, ymm8, ymm2
    vcvtdq2ps ymm8, ymm8
    vpermps ymm9, ymm3, ymmword ptr [r10 + 256]
    vfmadd231ps ymm0, ymm8, ymm9
    vpmaddubsw ymm7, ymm7, ymmword ptr [rdx + 288]
    vpmaddwd ymm7, ymm7, ymm2
    vcvtdq2ps ymm7, ymm7
    vpermps ymm8, ymm3, ymmword ptr [r10 + 260]
    vfmadd231ps ymm1, ymm7, ymm8
    # block 5
    vpand ymm7, ymm5, ymmword ptr [rip + {c5}]
    vpsllw ymm8, ymm6, 2
    vpand ymm8, ymm8, ymmword ptr [rip + {c6}]
    vpor ymm7, ymm7, ymm8
    vpmaddubsw ymm8, ymm7, ymmword ptr [rdx + 320]
    vpmaddwd ymm8, ymm8, ymm2
    vcvtdq2ps ymm8, ymm8
    vpermps ymm9, ymm3, ymmword ptr [r10 + 320]
    vfmadd231ps ymm0, ymm8, ymm9
    vpmaddubsw ymm7, ymm7, ymmword ptr [rdx + 352]
    vpmaddwd ymm7, ymm7, ymm2
    vcvtdq2ps ymm7, ymm7
    vpermps ymm8, ymm3, ymmword ptr [r10 + 324]
    vfmadd231ps ymm1, ymm7, ymm8
    # block 6
    vpsrlw ymm4, ymm4, 4
    vpand ymm4, ymm4, ymmword ptr [rip + {c5}]
    vpand ymm7, ymm6, ymmword ptr [rip + {c6}]
    vpor ymm4, ymm4, ymm7
    vpmaddubsw ymm7, ymm4, ymmword ptr [rdx + 384]
    vpmaddwd ymm7, ymm7, ymm2
    vcvtdq2ps ymm7, ymm7
    vpermps ymm8, ymm3, ymmword ptr [r10 + 384]
    vfmadd231ps ymm0, ymm7, ymm8
    vpmaddubsw ymm4, ymm4, ymmword ptr [rdx + 416]
    vpmaddwd ymm4, ymm4, ymm2
    vcvtdq2ps ymm4, ymm4
    vpermps ymm7, ymm3, ymmword ptr [r10 + 388]
    vfmadd231ps ymm1, ymm4, ymm7
    # block 7
    vpsrlw ymm4, ymm5, 4
    vpand ymm4, ymm4, ymmword ptr [rip + {c5}]
    vpsrlw ymm5, ymm6, 2
    vpand ymm5, ymm5, ymmword ptr [rip + {c6}]
    vpor ymm4, ymm4, ymm5
    vpmaddubsw ymm5, ymm4, ymmword ptr [rdx + 448]
    vpmaddwd ymm5, ymm5, ymm2
    vcvtdq2ps ymm5, ymm5
    vpermps ymm6, ymm3, ymmword ptr [r10 + 448]
    vfmadd231ps ymm0, ymm5, ymm6
    vpmaddubsw ymm4, ymm4, ymmword ptr [rdx + 480]
    vpmaddwd ymm4, ymm4, ymm2
    vcvtdq2ps ymm4, ymm4
    vpermps ymm2, ymm3, ymmword ptr [r10 + 452]
    vfmadd231ps ymm1, ymm4, ymm2
    vaddps ymm0, ymm0, ymmword ptr [rsp]
    vextractf128 xmm2, ymm0, 1
    vaddps xmm0, xmm0, xmm2
    vmovhlps xmm2, xmm0, xmm0
    vaddps xmm0, xmm0, xmm2
    vmovshdup xmm2, xmm0
    vaddss xmm0, xmm0, xmm2
    vmovss dword ptr [r9], xmm0
    vaddps ymm0, ymm1, ymmword ptr [rsp + 32]
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
# c0, aligned to 32:
#   00 00 00 00 00 00 00 00 01 00 00 00 01 00 00 00
#   00 00 00 00 00 00 00 00 01 00 00 00 01 00 00 00
# c1, aligned to 32:
#   02 00 00 00 02 00 00 00 03 00 00 00 03 00 00 00
#   02 00 00 00 02 00 00 00 03 00 00 00 03 00 00 00
# c2, aligned to 32:
#   04 00 00 00 04 00 00 00 05 00 00 00 05 00 00 00
#   04 00 00 00 04 00 00 00 05 00 00 00 05 00 00 00
# c3, aligned to 32:
#   06 00 00 00 06 00 00 00 07 00 00 00 07 00 00 00
#   06 00 00 00 06 00 00 00 07 00 00 00 07 00 00 00
# c4, aligned to 32:
#   00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
#   02 00 00 00 02 00 00 00 02 00 00 00 02 00 00 00
# c5, aligned to 32:
#   0f 0f 0f 0f 0f 0f 0f 0f 0f 0f 0f 0f 0f 0f 0f 0f
#   0f 0f 0f 0f 0f 0f 0f 0f 0f 0f 0f 0f 0f 0f 0f 0f
# c6, aligned to 32:
#   30 30 30 30 30 30 30 30 30 30 30 30 30 30 30 30
#   30 30 30 30 30 30 30 30 30 30 30 30 30 30 30 30
# c7, aligned to 32:
#   01 00 01 00 01 00 01 00 01 00 01 00 01 00 01 00
#   01 00 01 00 01 00 01 00 01 00 01 00 01 00 01 00
# c8, aligned to 32:
#   00 00 00 c2 00 00 00 c2 00 00 00 c2 00 00 00 c2
#   00 00 00 c2 00 00 00 c2 00 00 00 c2 00 00 00 c2
