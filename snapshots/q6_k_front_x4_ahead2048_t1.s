# q6_k_front_x4_ahead2048_t1(gpr, gpr, gpr, gpr, gpr, gpr) -> ()
# frame: 4096 bytes aligned to 2048, saves [rbx, r12]
# row=rdi sbs=rsi quants=rdx scales=rcx half_sums=r8 out=r9
# stride=rax sums1=r10 sums2=r11 sums3=rax acc0=ymm0 acc1=ymm1
# acc2=ymm2 acc3=ymm3 zero=ymm1,ymm2,ymm3,ymm4 cur=rbx nxt=r12 one16=ymm4
# lane=ymm5 half_d=ymm6,ymm10 d=ymm6,ymm10 d_times_scales=ymm7,ymm11 spread=ymm6,ymm9,ymm7 scale_products=ymm6,ymm9,ymm7
# activation_scales=ymm7,ymm10,ymm8,ymm11 offsets_lo=ymm6 offsets_hi=ymm7 off=ymm8 nibbles_even=ymm6 nibbles_odd=ymm7,ymm8
# bit_pairs=ymm8,ymm9 weights=ymm9,ymm6,ymm10 bit_pair=ymm10,ymm12,ymm9,ymm7,ymm11 products=ymm12,ymm9,ymm10,ymm6,ymm8,ymm11,ymm7 block_scale=ymm13,ymm12,ymm11,ymm10,ymm9,ymm8,ymm7,ymm4 sum=ymm0
# high=ymm4,ymm1 pair=ymm4,ymm1 result=ymm0 nothing=ymm0
    push rbx
    push r12
    push rbp
    mov rbp, rsp
    and rsp, -2048
    sub rsp, 4096
    test rsi, rsi
    jz 4f
    # constants
    mov rax, rsi
    shl rax, 6
    lea r10, [r8 + rax*1]
    lea r11, [r10 + rax*1]
    lea rax, [r11 + rax*1]
    vxorps xmm0, xmm0, xmm0
    vxorps xmm1, xmm1, xmm1
    vmovaps ymmword ptr [rsp], ymm1
    vxorps xmm1, xmm1, xmm1
    vxorps xmm2, xmm2, xmm2
    vmovaps ymmword ptr [rsp + 32], ymm2
    vxorps xmm2, xmm2, xmm2
    vxorps xmm3, xmm3, xmm3
    vmovaps ymmword ptr [rsp + 64], ymm3
    vxorps xmm3, xmm3, xmm3
    vxorps xmm4, xmm4, xmm4
    vmovaps ymmword ptr [rsp + 96], ymm4
    lea rbx, [rsp + 2048]
    lea r12, [rsp + 3072]
    vmovdqa ymm4, ymmword ptr [rip + {c7}]
    vmovdqa ymm5, ymmword ptr [rip + {c4}]
    # scales of superblock +0
    vpbroadcastw xmm6, word ptr [rdi]
    vcvtph2ps ymm6, xmm6
    vpmovsxbd ymm7, qword ptr [rdi + 2]
    vcvtdq2ps ymm7, ymm7
    vmulps ymm7, ymm7, ymm6
    vmovaps ymmword ptr [rbx + 512], ymm7
    vmulps ymm7, ymm7, ymmword ptr [rip + {c8}]
    vmovaps ymmword ptr [rbx + 576], ymm7
    vpmovsxbd ymm7, qword ptr [rdi + 10]
    vcvtdq2ps ymm7, ymm7
    vmulps ymm7, ymm7, ymm6
    vmovaps ymmword ptr [rbx + 544], ymm7
    vmulps ymm7, ymm7, ymmword ptr [rip + {c8}]
    vmovaps ymmword ptr [rbx + 608], ymm7
    vmovdqa ymm6, ymmword ptr [rip + {c0}]
    vpermps ymm6, ymm6, ymmword ptr [rbx + 512]
    vbroadcastf128 ymm7, xmmword ptr [rcx]
    vmulps ymm6, ymm6, ymm7
    vmovaps ymmword ptr [rbx], ymm6
    vmovdqa ymm6, ymmword ptr [rip + {c1}]
    vpermps ymm6, ymm6, ymmword ptr [rbx + 512]
    vbroadcastf128 ymm7, xmmword ptr [rcx + 16]
    vmulps ymm6, ymm6, ymm7
    vmovaps ymmword ptr [rbx + 64], ymm6
    vmovdqa ymm6, ymmword ptr [rip + {c2}]
    vpermps ymm6, ymm6, ymmword ptr [rbx + 512]
    vbroadcastf128 ymm7, xmmword ptr [rcx + 32]
    vmulps ymm6, ymm6, ymm7
    vmovaps ymmword ptr [rbx + 128], ymm6
    vmovdqa ymm6, ymmword ptr [rip + {c3}]
    vpermps ymm6, ymm6, ymmword ptr [rbx + 512]
    vbroadcastf128 ymm7, xmmword ptr [rcx + 48]
    vmulps ymm6, ymm6, ymm7
    vmovaps ymmword ptr [rbx + 192], ymm6
    vmovdqa ymm6, ymmword ptr [rip + {c0}]
    vpermps ymm6, ymm6, ymmword ptr [rbx + 544]
    vbroadcastf128 ymm7, xmmword ptr [rcx + 64]
    vmulps ymm6, ymm6, ymm7
    vmovaps ymmword ptr [rbx + 256], ymm6
    vmovdqa ymm6, ymmword ptr [rip + {c1}]
    vpermps ymm6, ymm6, ymmword ptr [rbx + 544]
    vbroadcastf128 ymm7, xmmword ptr [rcx + 80]
    vmulps ymm6, ymm6, ymm7
    vmovaps ymmword ptr [rbx + 320], ymm6
    vmovdqa ymm6, ymmword ptr [rip + {c2}]
    vpermps ymm6, ymm6, ymmword ptr [rbx + 544]
    vbroadcastf128 ymm7, xmmword ptr [rcx + 96]
    vmulps ymm6, ymm6, ymm7
    vmovaps ymmword ptr [rbx + 384], ymm6
    vmovdqa ymm6, ymmword ptr [rip + {c3}]
    vpermps ymm6, ymm6, ymmword ptr [rbx + 544]
    vbroadcastf128 ymm7, xmmword ptr [rcx + 112]
    vmulps ymm6, ymm6, ymm7
    vmovaps ymmword ptr [rbx + 448], ymm6
    dec rsi
    jz 3f
2:
    prefetcht1 byte ptr [rdi + 2048]
    prefetcht1 byte ptr [rdi + 2112]
    prefetcht1 byte ptr [rdi + 2176]
    prefetcht1 byte ptr [rdi + 2240]
    # offsets
    vmovdqa ymm6, ymmword ptr [rbx + 576]
    vmovdqa ymm7, ymmword ptr [rbx + 608]
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
    vmovdqu ymm6, ymmword ptr [rdi + 32]
    vmovdqu ymm7, ymmword ptr [rdi + 64]
    vmovdqu ymm8, ymmword ptr [rdi + 160]
    # block 0
    vpand ymm9, ymm6, ymmword ptr [rip + {c5}]
    vpsllw ymm10, ymm8, 4
    vpand ymm10, ymm10, ymmword ptr [rip + {c6}]
    vpor ymm9, ymm9, ymm10
    # scales of superblock +1
    vpbroadcastw xmm10, word ptr [rdi + 224]
    vcvtph2ps ymm10, xmm10
    vpmovsxbd ymm11, qword ptr [rdi + 226]
    vcvtdq2ps ymm11, ymm11
    vmulps ymm11, ymm11, ymm10
    vmovaps ymmword ptr [r12 + 512], ymm11
    vmulps ymm11, ymm11, ymmword ptr [rip + {c8}]
    vpmaddubsw ymm12, ymm9, ymmword ptr [rdx]
    vpmaddwd ymm12, ymm12, ymm4
    vcvtdq2ps ymm12, ymm12
    vpermps ymm13, ymm5, ymmword ptr [rbx]
    vfmadd231ps ymm0, ymm12, ymm13
    vpmaddubsw ymm12, ymm9, ymmword ptr [rdx + 32]
    vpmaddwd ymm12, ymm12, ymm4
    vcvtdq2ps ymm12, ymm12
    vpermps ymm13, ymm5, ymmword ptr [rbx + 4]
    vfmadd231ps ymm1, ymm12, ymm13
    vpmaddubsw ymm12, ymm9, ymmword ptr [rdx + 64]
    vpmaddwd ymm12, ymm12, ymm4
    vcvtdq2ps ymm12, ymm12
    vpermps ymm13, ymm5, ymmword ptr [rbx + 8]
    vfmadd231ps ymm2, ymm12, ymm13
    vpmaddubsw ymm9, ymm9, ymmword ptr [rdx + 96]
    vpmaddwd ymm9, ymm9, ymm4
    vcvtdq2ps ymm9, ymm9
    vpermps ymm12, ymm5, ymmword ptr [rbx + 12]
    vfmadd231ps ymm3, ymm9, ymm12
    # block 1
    vpand ymm9, ymm7, ymmword ptr [rip + {c5}]
    vpsllw ymm12, ymm8, 2
    vpand ymm12, ymm12, ymmword ptr [rip + {c6}]
    vpor ymm9, ymm9, ymm12
    vmovaps ymmword ptr [r12 + 576], ymm11
    vpmovsxbd ymm11, qword ptr [rdi + 234]
    vcvtdq2ps ymm11, ymm11
    vmulps ymm11, ymm11, ymm10
    vmovaps ymmword ptr [r12 + 544], ymm11
    vmulps ymm11, ymm11, ymmword ptr [rip + {c8}]
    vmovaps ymmword ptr [r12 + 608], ymm11
    vpmaddubsw ymm10, ymm9, ymmword ptr [rdx + 128]
    vpmaddwd ymm10, ymm10, ymm4
    vcvtdq2ps ymm10, ymm10
    vpermps ymm11, ymm5, ymmword ptr [rbx + 64]
    vfmadd231ps ymm0, ymm10, ymm11
    vpmaddubsw ymm10, ymm9, ymmword ptr [rdx + 160]
    vpmaddwd ymm10, ymm10, ymm4
    vcvtdq2ps ymm10, ymm10
    vpermps ymm11, ymm5, ymmword ptr [rbx + 68]
    vfmadd231ps ymm1, ymm10, ymm11
    vpmaddubsw ymm10, ymm9, ymmword ptr [rdx + 192]
    vpmaddwd ymm10, ymm10, ymm4
    vcvtdq2ps ymm10, ymm10
    vpermps ymm11, ymm5, ymmword ptr [rbx + 72]
    vfmadd231ps ymm2, ymm10, ymm11
    vpmaddubsw ymm9, ymm9, ymmword ptr [rdx + 224]
    vpmaddwd ymm9, ymm9, ymm4
    vcvtdq2ps ymm9, ymm9
    vpermps ymm10, ymm5, ymmword ptr [rbx + 76]
    vfmadd231ps ymm3, ymm9, ymm10
    # block 2
    vpsrlw ymm6, ymm6, 4
    vpand ymm6, ymm6, ymmword ptr [rip + {c5}]
    vpand ymm9, ymm8, ymmword ptr [rip + {c6}]
    vpor ymm6, ymm6, ymm9
    vmovdqa ymm9, ymmword ptr [rip + {c0}]
    vpermps ymm9, ymm9, ymmword ptr [r12 + 512]
    vbroadcastf128 ymm10, xmmword ptr [rcx + 128]
    vmulps ymm9, ymm9, ymm10
    vmovaps ymmword ptr [r12], ymm9
    vmovdqa ymm9, ymmword ptr [rip + {c1}]
    vpermps ymm9, ymm9, ymmword ptr [r12 + 512]
    vpmaddubsw ymm10, ymm6, ymmword ptr [rdx + 256]
    vpmaddwd ymm10, ymm10, ymm4
    vcvtdq2ps ymm10, ymm10
    vpermps ymm11, ymm5, ymmword ptr [rbx + 128]
    vfmadd231ps ymm0, ymm10, ymm11
    vpmaddubsw ymm10, ymm6, ymmword ptr [rdx + 288]
    vpmaddwd ymm10, ymm10, ymm4
    vcvtdq2ps ymm10, ymm10
    vpermps ymm11, ymm5, ymmword ptr [rbx + 132]
    vfmadd231ps ymm1, ymm10, ymm11
    vpmaddubsw ymm10, ymm6, ymmword ptr [rdx + 320]
    vpmaddwd ymm10, ymm10, ymm4
    vcvtdq2ps ymm10, ymm10
    vpermps ymm11, ymm5, ymmword ptr [rbx + 136]
    vfmadd231ps ymm2, ymm10, ymm11
    vpmaddubsw ymm6, ymm6, ymmword ptr [rdx + 352]
    vpmaddwd ymm6, ymm6, ymm4
    vcvtdq2ps ymm6, ymm6
    vpermps ymm10, ymm5, ymmword ptr [rbx + 140]
    vfmadd231ps ymm3, ymm6, ymm10
    # block 3
    vpsrlw ymm6, ymm7, 4
    vpand ymm6, ymm6, ymmword ptr [rip + {c5}]
    vpsrlw ymm7, ymm8, 2
    vpand ymm7, ymm7, ymmword ptr [rip + {c6}]
    vpor ymm6, ymm6, ymm7
    vbroadcastf128 ymm7, xmmword ptr [rcx + 144]
    vmulps ymm9, ymm9, ymm7
    vmovaps ymmword ptr [r12 + 64], ymm9
    vmovdqa ymm7, ymmword ptr [rip + {c2}]
    vpermps ymm7, ymm7, ymmword ptr [r12 + 512]
    vbroadcastf128 ymm8, xmmword ptr [rcx + 160]
    vmulps ymm7, ymm7, ymm8
    vpmaddubsw ymm8, ymm6, ymmword ptr [rdx + 384]
    vpmaddwd ymm8, ymm8, ymm4
    vcvtdq2ps ymm8, ymm8
    vpermps ymm9, ymm5, ymmword ptr [rbx + 192]
    vfmadd231ps ymm0, ymm8, ymm9
    vpmaddubsw ymm8, ymm6, ymmword ptr [rdx + 416]
    vpmaddwd ymm8, ymm8, ymm4
    vcvtdq2ps ymm8, ymm8
    vpermps ymm9, ymm5, ymmword ptr [rbx + 196]
    vfmadd231ps ymm1, ymm8, ymm9
    vpmaddubsw ymm8, ymm6, ymmword ptr [rdx + 448]
    vpmaddwd ymm8, ymm8, ymm4
    vcvtdq2ps ymm8, ymm8
    vpermps ymm9, ymm5, ymmword ptr [rbx + 200]
    vfmadd231ps ymm2, ymm8, ymm9
    vpmaddubsw ymm6, ymm6, ymmword ptr [rdx + 480]
    vpmaddwd ymm6, ymm6, ymm4
    vcvtdq2ps ymm6, ymm6
    vpermps ymm8, ymm5, ymmword ptr [rbx + 204]
    vfmadd231ps ymm3, ymm6, ymm8
    vmovdqu ymm6, ymmword ptr [rdi + 96]
    vmovdqu ymm8, ymmword ptr [rdi + 128]
    vmovdqu ymm9, ymmword ptr [rdi + 192]
    # block 4
    vpand ymm10, ymm6, ymmword ptr [rip + {c5}]
    vpsllw ymm11, ymm9, 4
    vpand ymm11, ymm11, ymmword ptr [rip + {c6}]
    vpor ymm10, ymm10, ymm11
    vmovaps ymmword ptr [r12 + 128], ymm7
    vmovdqa ymm7, ymmword ptr [rip + {c3}]
    vpermps ymm7, ymm7, ymmword ptr [r12 + 512]
    vbroadcastf128 ymm11, xmmword ptr [rcx + 176]
    vmulps ymm7, ymm7, ymm11
    vmovaps ymmword ptr [r12 + 192], ymm7
    vmovdqa ymm7, ymmword ptr [rip + {c0}]
    vpmaddubsw ymm11, ymm10, ymmword ptr [rdx + 512]
    vpmaddwd ymm11, ymm11, ymm4
    vcvtdq2ps ymm11, ymm11
    vpermps ymm12, ymm5, ymmword ptr [rbx + 256]
    vfmadd231ps ymm0, ymm11, ymm12
    vpmaddubsw ymm11, ymm10, ymmword ptr [rdx + 544]
    vpmaddwd ymm11, ymm11, ymm4
    vcvtdq2ps ymm11, ymm11
    vpermps ymm12, ymm5, ymmword ptr [rbx + 260]
    vfmadd231ps ymm1, ymm11, ymm12
    vpmaddubsw ymm11, ymm10, ymmword ptr [rdx + 576]
    vpmaddwd ymm11, ymm11, ymm4
    vcvtdq2ps ymm11, ymm11
    vpermps ymm12, ymm5, ymmword ptr [rbx + 264]
    vfmadd231ps ymm2, ymm11, ymm12
    vpmaddubsw ymm10, ymm10, ymmword ptr [rdx + 608]
    vpmaddwd ymm10, ymm10, ymm4
    vcvtdq2ps ymm10, ymm10
    vpermps ymm11, ymm5, ymmword ptr [rbx + 268]
    vfmadd231ps ymm3, ymm10, ymm11
    # block 5
    vpand ymm10, ymm8, ymmword ptr [rip + {c5}]
    vpsllw ymm11, ymm9, 2
    vpand ymm11, ymm11, ymmword ptr [rip + {c6}]
    vpor ymm10, ymm10, ymm11
    vpermps ymm7, ymm7, ymmword ptr [r12 + 544]
    vbroadcastf128 ymm11, xmmword ptr [rcx + 192]
    vmulps ymm7, ymm7, ymm11
    vmovaps ymmword ptr [r12 + 256], ymm7
    vmovdqa ymm7, ymmword ptr [rip + {c1}]
    vpermps ymm7, ymm7, ymmword ptr [r12 + 544]
    vbroadcastf128 ymm11, xmmword ptr [rcx + 208]
    vpmaddubsw ymm12, ymm10, ymmword ptr [rdx + 640]
    vpmaddwd ymm12, ymm12, ymm4
    vcvtdq2ps ymm12, ymm12
    vpermps ymm13, ymm5, ymmword ptr [rbx + 320]
    vfmadd231ps ymm0, ymm12, ymm13
    vpmaddubsw ymm12, ymm10, ymmword ptr [rdx + 672]
    vpmaddwd ymm12, ymm12, ymm4
    vcvtdq2ps ymm12, ymm12
    vpermps ymm13, ymm5, ymmword ptr [rbx + 324]
    vfmadd231ps ymm1, ymm12, ymm13
    vpmaddubsw ymm12, ymm10, ymmword ptr [rdx + 704]
    vpmaddwd ymm12, ymm12, ymm4
    vcvtdq2ps ymm12, ymm12
    vpermps ymm13, ymm5, ymmword ptr [rbx + 328]
    vfmadd231ps ymm2, ymm12, ymm13
    vpmaddubsw ymm10, ymm10, ymmword ptr [rdx + 736]
    vpmaddwd ymm10, ymm10, ymm4
    vcvtdq2ps ymm10, ymm10
    vpermps ymm12, ymm5, ymmword ptr [rbx + 332]
    vfmadd231ps ymm3, ymm10, ymm12
    # block 6
    vpsrlw ymm6, ymm6, 4
    vpand ymm6, ymm6, ymmword ptr [rip + {c5}]
    vpand ymm10, ymm9, ymmword ptr [rip + {c6}]
    vpor ymm6, ymm6, ymm10
    vmulps ymm7, ymm7, ymm11
    vmovaps ymmword ptr [r12 + 320], ymm7
    vmovdqa ymm7, ymmword ptr [rip + {c2}]
    vpermps ymm7, ymm7, ymmword ptr [r12 + 544]
    vbroadcastf128 ymm10, xmmword ptr [rcx + 224]
    vmulps ymm7, ymm7, ymm10
    vmovaps ymmword ptr [r12 + 384], ymm7
    vpmaddubsw ymm7, ymm6, ymmword ptr [rdx + 768]
    vpmaddwd ymm7, ymm7, ymm4
    vcvtdq2ps ymm7, ymm7
    vpermps ymm10, ymm5, ymmword ptr [rbx + 384]
    vfmadd231ps ymm0, ymm7, ymm10
    vpmaddubsw ymm7, ymm6, ymmword ptr [rdx + 800]
    vpmaddwd ymm7, ymm7, ymm4
    vcvtdq2ps ymm7, ymm7
    vpermps ymm10, ymm5, ymmword ptr [rbx + 388]
    vfmadd231ps ymm1, ymm7, ymm10
    vpmaddubsw ymm7, ymm6, ymmword ptr [rdx + 832]
    vpmaddwd ymm7, ymm7, ymm4
    vcvtdq2ps ymm7, ymm7
    vpermps ymm10, ymm5, ymmword ptr [rbx + 392]
    vfmadd231ps ymm2, ymm7, ymm10
    vpmaddubsw ymm6, ymm6, ymmword ptr [rdx + 864]
    vpmaddwd ymm6, ymm6, ymm4
    vcvtdq2ps ymm6, ymm6
    vpermps ymm7, ymm5, ymmword ptr [rbx + 396]
    vfmadd231ps ymm3, ymm6, ymm7
    # block 7
    vpsrlw ymm6, ymm8, 4
    vpand ymm6, ymm6, ymmword ptr [rip + {c5}]
    vpsrlw ymm7, ymm9, 2
    vpand ymm7, ymm7, ymmword ptr [rip + {c6}]
    vpor ymm6, ymm6, ymm7
    vmovdqa ymm7, ymmword ptr [rip + {c3}]
    vpermps ymm7, ymm7, ymmword ptr [r12 + 544]
    vbroadcastf128 ymm8, xmmword ptr [rcx + 240]
    vmulps ymm7, ymm7, ymm8
    vmovaps ymmword ptr [r12 + 448], ymm7
    vpmaddubsw ymm7, ymm6, ymmword ptr [rdx + 896]
    vpmaddwd ymm7, ymm7, ymm4
    vcvtdq2ps ymm7, ymm7
    vpermps ymm8, ymm5, ymmword ptr [rbx + 448]
    vfmadd231ps ymm0, ymm7, ymm8
    vpmaddubsw ymm7, ymm6, ymmword ptr [rdx + 928]
    vpmaddwd ymm7, ymm7, ymm4
    vcvtdq2ps ymm7, ymm7
    vpermps ymm8, ymm5, ymmword ptr [rbx + 452]
    vfmadd231ps ymm1, ymm7, ymm8
    vpmaddubsw ymm7, ymm6, ymmword ptr [rdx + 960]
    vpmaddwd ymm7, ymm7, ymm4
    vcvtdq2ps ymm7, ymm7
    vpermps ymm8, ymm5, ymmword ptr [rbx + 456]
    vfmadd231ps ymm2, ymm7, ymm8
    vpmaddubsw ymm6, ymm6, ymmword ptr [rdx + 992]
    vpmaddwd ymm6, ymm6, ymm4
    vcvtdq2ps ymm6, ymm6
    vpermps ymm7, ymm5, ymmword ptr [rbx + 460]
    vfmadd231ps ymm3, ymm6, ymm7
    add rdi, 224
    add rdx, 1024
    add rcx, 128
    add r8, 64
    add r10, 64
    add r11, 64
    add rax, 64
    xor rbx, 1024
    xor r12, 1024
    dec rsi
    jnz 2b
3:
    # offsets
    vmovdqa ymm6, ymmword ptr [rbx + 576]
    vmovdqa ymm7, ymmword ptr [rbx + 608]
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
    vmovdqu ymm6, ymmword ptr [rdi + 32]
    vmovdqu ymm7, ymmword ptr [rdi + 64]
    vmovdqu ymm8, ymmword ptr [rdi + 160]
    # block 0
    vpand ymm9, ymm6, ymmword ptr [rip + {c5}]
    vpsllw ymm10, ymm8, 4
    vpand ymm10, ymm10, ymmword ptr [rip + {c6}]
    vpor ymm9, ymm9, ymm10
    vpmaddubsw ymm10, ymm9, ymmword ptr [rdx]
    vpmaddwd ymm10, ymm10, ymm4
    vcvtdq2ps ymm10, ymm10
    vpermps ymm11, ymm5, ymmword ptr [rbx]
    vfmadd231ps ymm0, ymm10, ymm11
    vpmaddubsw ymm10, ymm9, ymmword ptr [rdx + 32]
    vpmaddwd ymm10, ymm10, ymm4
    vcvtdq2ps ymm10, ymm10
    vpermps ymm11, ymm5, ymmword ptr [rbx + 4]
    vfmadd231ps ymm1, ymm10, ymm11
    vpmaddubsw ymm10, ymm9, ymmword ptr [rdx + 64]
    vpmaddwd ymm10, ymm10, ymm4
    vcvtdq2ps ymm10, ymm10
    vpermps ymm11, ymm5, ymmword ptr [rbx + 8]
    vfmadd231ps ymm2, ymm10, ymm11
    vpmaddubsw ymm9, ymm9, ymmword ptr [rdx + 96]
    vpmaddwd ymm9, ymm9, ymm4
    vcvtdq2ps ymm9, ymm9
    vpermps ymm10, ymm5, ymmword ptr [rbx + 12]
    vfmadd231ps ymm3, ymm9, ymm10
    # block 1
    vpand ymm9, ymm7, ymmword ptr [rip + {c5}]
    vpsllw ymm10, ymm8, 2
    vpand ymm10, ymm10, ymmword ptr [rip + {c6}]
    vpor ymm9, ymm9, ymm10
    vpmaddubsw ymm10, ymm9, ymmword ptr [rdx + 128]
    vpmaddwd ymm10, ymm10, ymm4
    vcvtdq2ps ymm10, ymm10
    vpermps ymm11, ymm5, ymmword ptr [rbx + 64]
    vfmadd231ps ymm0, ymm10, ymm11
    vpmaddubsw ymm10, ymm9, ymmword ptr [rdx + 160]
    vpmaddwd ymm10, ymm10, ymm4
    vcvtdq2ps ymm10, ymm10
    vpermps ymm11, ymm5, ymmword ptr [rbx + 68]
    vfmadd231ps ymm1, ymm10, ymm11
    vpmaddubsw ymm10, ymm9, ymmword ptr [rdx + 192]
    vpmaddwd ymm10, ymm10, ymm4
    vcvtdq2ps ymm10, ymm10
    vpermps ymm11, ymm5, ymmword ptr [rbx + 72]
    vfmadd231ps ymm2, ymm10, ymm11
    vpmaddubsw ymm9, ymm9, ymmword ptr [rdx + 224]
    vpmaddwd ymm9, ymm9, ymm4
    vcvtdq2ps ymm9, ymm9
    vpermps ymm10, ymm5, ymmword ptr [rbx + 76]
    vfmadd231ps ymm3, ymm9, ymm10
    # block 2
    vpsrlw ymm6, ymm6, 4
    vpand ymm6, ymm6, ymmword ptr [rip + {c5}]
    vpand ymm9, ymm8, ymmword ptr [rip + {c6}]
    vpor ymm6, ymm6, ymm9
    vpmaddubsw ymm9, ymm6, ymmword ptr [rdx + 256]
    vpmaddwd ymm9, ymm9, ymm4
    vcvtdq2ps ymm9, ymm9
    vpermps ymm10, ymm5, ymmword ptr [rbx + 128]
    vfmadd231ps ymm0, ymm9, ymm10
    vpmaddubsw ymm9, ymm6, ymmword ptr [rdx + 288]
    vpmaddwd ymm9, ymm9, ymm4
    vcvtdq2ps ymm9, ymm9
    vpermps ymm10, ymm5, ymmword ptr [rbx + 132]
    vfmadd231ps ymm1, ymm9, ymm10
    vpmaddubsw ymm9, ymm6, ymmword ptr [rdx + 320]
    vpmaddwd ymm9, ymm9, ymm4
    vcvtdq2ps ymm9, ymm9
    vpermps ymm10, ymm5, ymmword ptr [rbx + 136]
    vfmadd231ps ymm2, ymm9, ymm10
    vpmaddubsw ymm6, ymm6, ymmword ptr [rdx + 352]
    vpmaddwd ymm6, ymm6, ymm4
    vcvtdq2ps ymm6, ymm6
    vpermps ymm9, ymm5, ymmword ptr [rbx + 140]
    vfmadd231ps ymm3, ymm6, ymm9
    # block 3
    vpsrlw ymm6, ymm7, 4
    vpand ymm6, ymm6, ymmword ptr [rip + {c5}]
    vpsrlw ymm7, ymm8, 2
    vpand ymm7, ymm7, ymmword ptr [rip + {c6}]
    vpor ymm6, ymm6, ymm7
    vpmaddubsw ymm7, ymm6, ymmword ptr [rdx + 384]
    vpmaddwd ymm7, ymm7, ymm4
    vcvtdq2ps ymm7, ymm7
    vpermps ymm8, ymm5, ymmword ptr [rbx + 192]
    vfmadd231ps ymm0, ymm7, ymm8
    vpmaddubsw ymm7, ymm6, ymmword ptr [rdx + 416]
    vpmaddwd ymm7, ymm7, ymm4
    vcvtdq2ps ymm7, ymm7
    vpermps ymm8, ymm5, ymmword ptr [rbx + 196]
    vfmadd231ps ymm1, ymm7, ymm8
    vpmaddubsw ymm7, ymm6, ymmword ptr [rdx + 448]
    vpmaddwd ymm7, ymm7, ymm4
    vcvtdq2ps ymm7, ymm7
    vpermps ymm8, ymm5, ymmword ptr [rbx + 200]
    vfmadd231ps ymm2, ymm7, ymm8
    vpmaddubsw ymm6, ymm6, ymmword ptr [rdx + 480]
    vpmaddwd ymm6, ymm6, ymm4
    vcvtdq2ps ymm6, ymm6
    vpermps ymm7, ymm5, ymmword ptr [rbx + 204]
    vfmadd231ps ymm3, ymm6, ymm7
    vmovdqu ymm6, ymmword ptr [rdi + 96]
    vmovdqu ymm7, ymmword ptr [rdi + 128]
    vmovdqu ymm8, ymmword ptr [rdi + 192]
    # block 4
    vpand ymm9, ymm6, ymmword ptr [rip + {c5}]
    vpsllw ymm10, ymm8, 4
    vpand ymm10, ymm10, ymmword ptr [rip + {c6}]
    vpor ymm9, ymm9, ymm10
    vpmaddubsw ymm10, ymm9, ymmword ptr [rdx + 512]
    vpmaddwd ymm10, ymm10, ymm4
    vcvtdq2ps ymm10, ymm10
    vpermps ymm11, ymm5, ymmword ptr [rbx + 256]
    vfmadd231ps ymm0, ymm10, ymm11
    vpmaddubsw ymm10, ymm9, ymmword ptr [rdx + 544]
    vpmaddwd ymm10, ymm10, ymm4
    vcvtdq2ps ymm10, ymm10
    vpermps ymm11, ymm5, ymmword ptr [rbx + 260]
    vfmadd231ps ymm1, ymm10, ymm11
    vpmaddubsw ymm10, ymm9, ymmword ptr [rdx + 576]
    vpmaddwd ymm10, ymm10, ymm4
    vcvtdq2ps ymm10, ymm10
    vpermps ymm11, ymm5, ymmword ptr [rbx + 264]
    vfmadd231ps ymm2, ymm10, ymm11
    vpmaddubsw ymm9, ymm9, ymmword ptr [rdx + 608]
    vpmaddwd ymm9, ymm9, ymm4
    vcvtdq2ps ymm9, ymm9
    vpermps ymm10, ymm5, ymmword ptr [rbx + 268]
    vfmadd231ps ymm3, ymm9, ymm10
    # block 5
    vpand ymm9, ymm7, ymmword ptr [rip + {c5}]
    vpsllw ymm10, ymm8, 2
    vpand ymm10, ymm10, ymmword ptr [rip + {c6}]
    vpor ymm9, ymm9, ymm10
    vpmaddubsw ymm10, ymm9, ymmword ptr [rdx + 640]
    vpmaddwd ymm10, ymm10, ymm4
    vcvtdq2ps ymm10, ymm10
    vpermps ymm11, ymm5, ymmword ptr [rbx + 320]
    vfmadd231ps ymm0, ymm10, ymm11
    vpmaddubsw ymm10, ymm9, ymmword ptr [rdx + 672]
    vpmaddwd ymm10, ymm10, ymm4
    vcvtdq2ps ymm10, ymm10
    vpermps ymm11, ymm5, ymmword ptr [rbx + 324]
    vfmadd231ps ymm1, ymm10, ymm11
    vpmaddubsw ymm10, ymm9, ymmword ptr [rdx + 704]
    vpmaddwd ymm10, ymm10, ymm4
    vcvtdq2ps ymm10, ymm10
    vpermps ymm11, ymm5, ymmword ptr [rbx + 328]
    vfmadd231ps ymm2, ymm10, ymm11
    vpmaddubsw ymm9, ymm9, ymmword ptr [rdx + 736]
    vpmaddwd ymm9, ymm9, ymm4
    vcvtdq2ps ymm9, ymm9
    vpermps ymm10, ymm5, ymmword ptr [rbx + 332]
    vfmadd231ps ymm3, ymm9, ymm10
    # block 6
    vpsrlw ymm6, ymm6, 4
    vpand ymm6, ymm6, ymmword ptr [rip + {c5}]
    vpand ymm9, ymm8, ymmword ptr [rip + {c6}]
    vpor ymm6, ymm6, ymm9
    vpmaddubsw ymm9, ymm6, ymmword ptr [rdx + 768]
    vpmaddwd ymm9, ymm9, ymm4
    vcvtdq2ps ymm9, ymm9
    vpermps ymm10, ymm5, ymmword ptr [rbx + 384]
    vfmadd231ps ymm0, ymm9, ymm10
    vpmaddubsw ymm9, ymm6, ymmword ptr [rdx + 800]
    vpmaddwd ymm9, ymm9, ymm4
    vcvtdq2ps ymm9, ymm9
    vpermps ymm10, ymm5, ymmword ptr [rbx + 388]
    vfmadd231ps ymm1, ymm9, ymm10
    vpmaddubsw ymm9, ymm6, ymmword ptr [rdx + 832]
    vpmaddwd ymm9, ymm9, ymm4
    vcvtdq2ps ymm9, ymm9
    vpermps ymm10, ymm5, ymmword ptr [rbx + 392]
    vfmadd231ps ymm2, ymm9, ymm10
    vpmaddubsw ymm6, ymm6, ymmword ptr [rdx + 864]
    vpmaddwd ymm6, ymm6, ymm4
    vcvtdq2ps ymm6, ymm6
    vpermps ymm9, ymm5, ymmword ptr [rbx + 396]
    vfmadd231ps ymm3, ymm6, ymm9
    # block 7
    vpsrlw ymm6, ymm7, 4
    vpand ymm6, ymm6, ymmword ptr [rip + {c5}]
    vpsrlw ymm7, ymm8, 2
    vpand ymm7, ymm7, ymmword ptr [rip + {c6}]
    vpor ymm6, ymm6, ymm7
    vpmaddubsw ymm7, ymm6, ymmword ptr [rdx + 896]
    vpmaddwd ymm7, ymm7, ymm4
    vcvtdq2ps ymm7, ymm7
    vpermps ymm8, ymm5, ymmword ptr [rbx + 448]
    vfmadd231ps ymm0, ymm7, ymm8
    vpmaddubsw ymm7, ymm6, ymmword ptr [rdx + 928]
    vpmaddwd ymm7, ymm7, ymm4
    vcvtdq2ps ymm7, ymm7
    vpermps ymm8, ymm5, ymmword ptr [rbx + 452]
    vfmadd231ps ymm1, ymm7, ymm8
    vpmaddubsw ymm7, ymm6, ymmword ptr [rdx + 960]
    vpmaddwd ymm7, ymm7, ymm4
    vcvtdq2ps ymm7, ymm7
    vpermps ymm8, ymm5, ymmword ptr [rbx + 456]
    vfmadd231ps ymm2, ymm7, ymm8
    vpmaddubsw ymm6, ymm6, ymmword ptr [rdx + 992]
    vpmaddwd ymm6, ymm6, ymm4
    vcvtdq2ps ymm6, ymm6
    vpermps ymm4, ymm5, ymmword ptr [rbx + 460]
    vfmadd231ps ymm3, ymm6, ymm4
    vaddps ymm0, ymm0, ymmword ptr [rsp]
    vextractf128 xmm4, ymm0, 1
    vaddps xmm0, xmm0, xmm4
    vmovhlps xmm4, xmm0, xmm0
    vaddps xmm0, xmm0, xmm4
    vmovshdup xmm4, xmm0
    vaddss xmm0, xmm0, xmm4
    vmovss dword ptr [r9], xmm0
    vaddps ymm0, ymm1, ymmword ptr [rsp + 32]
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vmovhlps xmm1, xmm0, xmm0
    vaddps xmm0, xmm0, xmm1
    vmovshdup xmm1, xmm0
    vaddss xmm0, xmm0, xmm1
    vmovss dword ptr [r9 + 4], xmm0
    vaddps ymm0, ymm2, ymmword ptr [rsp + 64]
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vmovhlps xmm1, xmm0, xmm0
    vaddps xmm0, xmm0, xmm1
    vmovshdup xmm1, xmm0
    vaddss xmm0, xmm0, xmm1
    vmovss dword ptr [r9 + 8], xmm0
    vaddps ymm0, ymm3, ymmword ptr [rsp + 96]
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
# c0, aligned to 32:
#   00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
#   01 00 00 00 01 00 00 00 01 00 00 00 01 00 00 00
# c1, aligned to 32:
#   02 00 00 00 02 00 00 00 02 00 00 00 02 00 00 00
#   03 00 00 00 03 00 00 00 03 00 00 00 03 00 00 00
# c2, aligned to 32:
#   04 00 00 00 04 00 00 00 04 00 00 00 04 00 00 00
#   05 00 00 00 05 00 00 00 05 00 00 00 05 00 00 00
# c3, aligned to 32:
#   06 00 00 00 06 00 00 00 06 00 00 00 06 00 00 00
#   07 00 00 00 07 00 00 00 07 00 00 00 07 00 00 00
# c4, aligned to 32:
#   00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
#   04 00 00 00 04 00 00 00 04 00 00 00 04 00 00 00
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
