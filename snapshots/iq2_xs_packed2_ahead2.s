# iq2_xs_packed2_ahead2(gpr, gpr, gpr, gpr, gpr, gpr) -> ()
# frame: 256 bytes aligned to 256, saves [rbx, r12, r13]
# row=rdi sbs=rsi quants=rdx scales=rcx grid=r8 out=r9
# spread=ymm0 bit=ymm1 max8=ymm2 one16=ymm3 acc0=ymm4 acc1=ymm5
# cur=rax nxt=r10 p0a=ymm8 p0b=ymm6 p1a=ymm10 p1b=ymm7
# p2a=ymm12 p2b=ymm9 p3a=ymm13 p3b=ymm10 p4a=ymm14 p4b=ymm11
# p5a=ymm12 p5b=ymm9 p6a=ymm11 p6b=ymm7 p7a=ymm12 p7b=ymm9
# d=ymm6 steps=ymm7,ymm8 low_scales=ymm8,ymm7 high_scales=ymm7,ymm8 pair_scale=ymm9,ymm8,ymm6,ymm7 index0=rbx,rsi
# index1=r12,r10 index2=r13,r11 index3=r11,rcx packed_indices=r11,rcx levels=ymm6,ymm7,ymm9,ymm10,ymm11 rows=ymm7,ymm9,ymm11,ymm12,ymm10,ymm8
# signs=ymm7,ymm9,ymm11,ymm12,ymm10,ymm8 x=ymm8,ymm9,ymm10,ymm11,ymm12,ymm13,ymm14,ymm15,ymm0 lanes=ymm8,ymm6,ymm10,ymm7,ymm11,ymm9,ymm12,ymm0 low_scale=ymm9,ymm8,ymm11,ymm10,ymm12,ymm13,ymm7,ymm1 high_scale=ymm11,ymm9,ymm13,ymm14,ymm12,ymm15,ymm10,ymm8,ymm2 scale=ymm9,ymm8,ymm11,ymm10,ymm12,ymm13,ymm7,ymm1
# zero=ymm0 high=ymm0 sum=ymm0 pair=ymm1 result=ymm0 nothing=ymm0
    push rbx
    push r12
    push r13
    push rbp
    mov rbp, rsp
    and rsp, -256
    sub rsp, 256
    test rsi, rsi
    jz 4f
    # constants
    vmovdqa ymm0, ymmword ptr [rip + {c0}]
    vpbroadcastq ymm1, qword ptr [rip + {c1}]
    vpbroadcastd ymm2, dword ptr [rip + {c2}]
    vpbroadcastd ymm3, dword ptr [rip + {c3}]
    vxorps xmm4, xmm4, xmm4
    vxorps xmm5, xmm5, xmm5
    lea rax, [rsp]
    lea r10, [rsp + 128]
    # pair scales of superblock +0
    vpbroadcastw xmm6, word ptr [rdi]
    vcvtph2ps ymm6, xmm6
    vpmovzxbd ymm7, qword ptr [rdi + 70]
    vpermq ymm7, ymm7, 0xd8
    vpand ymm8, ymm7, ymmword ptr [rip + {c4}]
    vpsrld ymm7, ymm7, 4
    vpaddd ymm8, ymm8, ymm8
    vpor ymm8, ymm8, ymmword ptr [rip + {c5}]
    vcvtdq2ps ymm8, ymm8
    vmulps ymm8, ymm6, ymm8
    vmulps ymm8, ymm8, ymmword ptr [rip + {c6}]
    vshufps ymm9, ymm8, ymm8, 0x50
    vmulps ymm9, ymm9, ymmword ptr [rcx]
    vmovaps ymmword ptr [rax], ymm9
    vshufps ymm8, ymm8, ymm8, 0xfa
    vmulps ymm8, ymm8, ymmword ptr [rcx + 32]
    vmovaps ymmword ptr [rax + 32], ymm8
    vpaddd ymm7, ymm7, ymm7
    vpor ymm7, ymm7, ymmword ptr [rip + {c5}]
    vcvtdq2ps ymm7, ymm7
    vmulps ymm7, ymm6, ymm7
    vmulps ymm7, ymm7, ymmword ptr [rip + {c6}]
    vshufps ymm6, ymm7, ymm7, 0x50
    vmulps ymm6, ymm6, ymmword ptr [rcx]
    vmovaps ymmword ptr [rax + 64], ymm6
    vshufps ymm6, ymm7, ymm7, 0xfa
    vmulps ymm6, ymm6, ymmword ptr [rcx + 32]
    vmovaps ymmword ptr [rax + 96], ymm6
    # products of group 0 of superblock +0
    mov r11, qword ptr [rdi + 2]
    mov rbx, r11
    and rbx, 511
    mov r12, r11
    shr r12, 9
    and r12, 511
    mov r13, r11
    shr r13, 18
    and r13, 511
    mov r11, r11
    shr r11, 27
    and r11, 511
    vmovq xmm6, qword ptr [r8 + rbx*8]
    vmovhps xmm6, xmm6, qword ptr [r8 + r12*8]
    vmovq xmm7, qword ptr [r8 + r13*8]
    vmovhps xmm7, xmm7, qword ptr [r8 + r11*8]
    vinserti128 ymm6, ymm6, xmm7, 1
    vpbroadcastd ymm7, dword ptr [rdi + 38]
    vpshufb ymm7, ymm7, ymm0
    vpand ymm7, ymm7, ymm1
    vpaddb ymm7, ymm7, ymm2
    vmovdqa ymm8, ymmword ptr [rdx]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm8, ymm6, ymm8
    vmovdqa ymm9, ymmword ptr [rdx + 32]
    vpsignb ymm9, ymm9, ymm7
    vpmaddubsw ymm6, ymm6, ymm9
    # products of group 1 of superblock +0
    mov r11, qword ptr [rdi + 6]
    shr r11, 4
    mov rbx, r11
    and rbx, 511
    mov r12, r11
    shr r12, 9
    and r12, 511
    mov r13, r11
    shr r13, 18
    and r13, 511
    mov r11, r11
    shr r11, 27
    and r11, 511
    vmovq xmm7, qword ptr [r8 + rbx*8]
    vmovhps xmm7, xmm7, qword ptr [r8 + r12*8]
    vmovq xmm9, qword ptr [r8 + r13*8]
    vmovhps xmm9, xmm9, qword ptr [r8 + r11*8]
    vinserti128 ymm7, ymm7, xmm9, 1
    vpbroadcastd ymm9, dword ptr [rdi + 42]
    vpshufb ymm9, ymm9, ymm0
    vpand ymm9, ymm9, ymm1
    vpaddb ymm9, ymm9, ymm2
    vmovdqa ymm10, ymmword ptr [rdx + 64]
    vpsignb ymm10, ymm10, ymm9
    vpmaddubsw ymm10, ymm7, ymm10
    vmovdqa ymm11, ymmword ptr [rdx + 96]
    vpsignb ymm11, ymm11, ymm9
    vpmaddubsw ymm7, ymm7, ymm11
    dec rsi
    jz 3f
2:
    # sums of group 0
    vpmaddwd ymm8, ymm8, ymm3
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm9, dword ptr [rax]
    vbroadcastss ymm11, dword ptr [rax + 64]
    vpblendd ymm9, ymm9, ymm11, 0xf0
    vfmadd231ps ymm4, ymm8, ymm9
    vpmaddwd ymm6, ymm6, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm8, dword ptr [rax + 4]
    vbroadcastss ymm9, dword ptr [rax + 68]
    vpblendd ymm8, ymm8, ymm9, 0xf0
    vfmadd231ps ymm5, ymm6, ymm8
    # pair scales of superblock +1
    vpbroadcastw xmm6, word ptr [rdi + 78]
    vcvtph2ps ymm6, xmm6
    vpmovzxbd ymm8, qword ptr [rdi + 148]
    vpermq ymm8, ymm8, 0xd8
    # products of group 2 of superblock +0
    mov r11, qword ptr [rdi + 11]
    mov rbx, r11
    and rbx, 511
    mov r12, r11
    shr r12, 9
    and r12, 511
    mov r13, r11
    shr r13, 18
    and r13, 511
    mov r11, r11
    shr r11, 27
    and r11, 511
    vmovq xmm9, qword ptr [r8 + rbx*8]
    vmovhps xmm9, xmm9, qword ptr [r8 + r12*8]
    vmovq xmm11, qword ptr [r8 + r13*8]
    vmovhps xmm11, xmm11, qword ptr [r8 + r11*8]
    vinserti128 ymm9, ymm9, xmm11, 1
    vpbroadcastd ymm11, dword ptr [rdi + 46]
    vpshufb ymm11, ymm11, ymm0
    vpand ymm11, ymm11, ymm1
    vpaddb ymm11, ymm11, ymm2
    vmovdqa ymm12, ymmword ptr [rdx + 128]
    vpsignb ymm12, ymm12, ymm11
    vpmaddubsw ymm12, ymm9, ymm12
    vmovdqa ymm13, ymmword ptr [rdx + 160]
    vpsignb ymm13, ymm13, ymm11
    vpmaddubsw ymm9, ymm9, ymm13
    # sums of group 1
    vpmaddwd ymm10, ymm10, ymm3
    vcvtdq2ps ymm10, ymm10
    vbroadcastss ymm11, dword ptr [rax + 8]
    vbroadcastss ymm13, dword ptr [rax + 72]
    vpblendd ymm11, ymm11, ymm13, 0xf0
    vfmadd231ps ymm4, ymm10, ymm11
    vpmaddwd ymm7, ymm7, ymm3
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm10, dword ptr [rax + 12]
    vbroadcastss ymm11, dword ptr [rax + 76]
    vpblendd ymm10, ymm10, ymm11, 0xf0
    vfmadd231ps ymm5, ymm7, ymm10
    vpand ymm7, ymm8, ymmword ptr [rip + {c4}]
    vpsrld ymm8, ymm8, 4
    vpaddd ymm7, ymm7, ymm7
    vpor ymm7, ymm7, ymmword ptr [rip + {c5}]
    # products of group 3 of superblock +0
    mov r11, qword ptr [rdi + 15]
    shr r11, 4
    mov rbx, r11
    and rbx, 511
    mov r12, r11
    shr r12, 9
    and r12, 511
    mov r13, r11
    shr r13, 18
    and r13, 511
    mov r11, r11
    shr r11, 27
    and r11, 511
    vmovq xmm10, qword ptr [r8 + rbx*8]
    vmovhps xmm10, xmm10, qword ptr [r8 + r12*8]
    vmovq xmm11, qword ptr [r8 + r13*8]
    vmovhps xmm11, xmm11, qword ptr [r8 + r11*8]
    vinserti128 ymm10, ymm10, xmm11, 1
    vpbroadcastd ymm11, dword ptr [rdi + 50]
    vpshufb ymm11, ymm11, ymm0
    vpand ymm11, ymm11, ymm1
    vpaddb ymm11, ymm11, ymm2
    vmovdqa ymm13, ymmword ptr [rdx + 192]
    vpsignb ymm13, ymm13, ymm11
    vpmaddubsw ymm13, ymm10, ymm13
    vmovdqa ymm14, ymmword ptr [rdx + 224]
    vpsignb ymm14, ymm14, ymm11
    vpmaddubsw ymm10, ymm10, ymm14
    # sums of group 2
    vpmaddwd ymm11, ymm12, ymm3
    vcvtdq2ps ymm11, ymm11
    vbroadcastss ymm12, dword ptr [rax + 16]
    vbroadcastss ymm14, dword ptr [rax + 80]
    vpblendd ymm12, ymm12, ymm14, 0xf0
    vfmadd231ps ymm4, ymm11, ymm12
    vpmaddwd ymm9, ymm9, ymm3
    vcvtdq2ps ymm9, ymm9
    vbroadcastss ymm11, dword ptr [rax + 20]
    vbroadcastss ymm12, dword ptr [rax + 84]
    vpblendd ymm11, ymm11, ymm12, 0xf0
    vfmadd231ps ymm5, ymm9, ymm11
    vcvtdq2ps ymm7, ymm7
    vmulps ymm7, ymm6, ymm7
    vmulps ymm7, ymm7, ymmword ptr [rip + {c6}]
    vshufps ymm9, ymm7, ymm7, 0x50
    # products of group 4 of superblock +0
    mov r11, qword ptr [rdi + 20]
    mov rbx, r11
    and rbx, 511
    mov r12, r11
    shr r12, 9
    and r12, 511
    mov r13, r11
    shr r13, 18
    and r13, 511
    mov r11, r11
    shr r11, 27
    and r11, 511
    vmovq xmm11, qword ptr [r8 + rbx*8]
    vmovhps xmm11, xmm11, qword ptr [r8 + r12*8]
    vmovq xmm12, qword ptr [r8 + r13*8]
    vmovhps xmm12, xmm12, qword ptr [r8 + r11*8]
    vinserti128 ymm11, ymm11, xmm12, 1
    vpbroadcastd ymm12, dword ptr [rdi + 54]
    vpshufb ymm12, ymm12, ymm0
    vpand ymm12, ymm12, ymm1
    vpaddb ymm12, ymm12, ymm2
    vmovdqa ymm14, ymmword ptr [rdx + 256]
    vpsignb ymm14, ymm14, ymm12
    vpmaddubsw ymm14, ymm11, ymm14
    vmovdqa ymm15, ymmword ptr [rdx + 288]
    vpsignb ymm15, ymm15, ymm12
    vpmaddubsw ymm11, ymm11, ymm15
    # sums of group 3
    vpmaddwd ymm12, ymm13, ymm3
    vcvtdq2ps ymm12, ymm12
    vbroadcastss ymm13, dword ptr [rax + 24]
    vbroadcastss ymm15, dword ptr [rax + 88]
    vpblendd ymm13, ymm13, ymm15, 0xf0
    vfmadd231ps ymm4, ymm12, ymm13
    vpmaddwd ymm10, ymm10, ymm3
    vcvtdq2ps ymm10, ymm10
    vbroadcastss ymm12, dword ptr [rax + 28]
    vbroadcastss ymm13, dword ptr [rax + 92]
    vpblendd ymm12, ymm12, ymm13, 0xf0
    vfmadd231ps ymm5, ymm10, ymm12
    vmulps ymm9, ymm9, ymmword ptr [rcx + 64]
    vmovaps ymmword ptr [r10], ymm9
    vshufps ymm7, ymm7, ymm7, 0xfa
    vmulps ymm7, ymm7, ymmword ptr [rcx + 96]
    # products of group 5 of superblock +0
    mov r11, qword ptr [rdi + 24]
    shr r11, 4
    mov rbx, r11
    and rbx, 511
    mov r12, r11
    shr r12, 9
    and r12, 511
    mov r13, r11
    shr r13, 18
    and r13, 511
    mov r11, r11
    shr r11, 27
    and r11, 511
    vmovq xmm9, qword ptr [r8 + rbx*8]
    vmovhps xmm9, xmm9, qword ptr [r8 + r12*8]
    vmovq xmm10, qword ptr [r8 + r13*8]
    vmovhps xmm10, xmm10, qword ptr [r8 + r11*8]
    vinserti128 ymm9, ymm9, xmm10, 1
    vpbroadcastd ymm10, dword ptr [rdi + 58]
    vpshufb ymm10, ymm10, ymm0
    vpand ymm10, ymm10, ymm1
    vpaddb ymm10, ymm10, ymm2
    vmovdqa ymm12, ymmword ptr [rdx + 320]
    vpsignb ymm12, ymm12, ymm10
    vpmaddubsw ymm12, ymm9, ymm12
    vmovdqa ymm13, ymmword ptr [rdx + 352]
    vpsignb ymm13, ymm13, ymm10
    vpmaddubsw ymm9, ymm9, ymm13
    # sums of group 4
    vpmaddwd ymm10, ymm14, ymm3
    vcvtdq2ps ymm10, ymm10
    vbroadcastss ymm13, dword ptr [rax + 32]
    vbroadcastss ymm14, dword ptr [rax + 96]
    vpblendd ymm13, ymm13, ymm14, 0xf0
    vfmadd231ps ymm4, ymm10, ymm13
    vpmaddwd ymm10, ymm11, ymm3
    vcvtdq2ps ymm10, ymm10
    vbroadcastss ymm11, dword ptr [rax + 36]
    vbroadcastss ymm13, dword ptr [rax + 100]
    vpblendd ymm11, ymm11, ymm13, 0xf0
    vfmadd231ps ymm5, ymm10, ymm11
    vmovaps ymmword ptr [r10 + 32], ymm7
    vpaddd ymm8, ymm8, ymm8
    vpor ymm8, ymm8, ymmword ptr [rip + {c5}]
    vcvtdq2ps ymm8, ymm8
    # products of group 6 of superblock +0
    mov r11, qword ptr [rdi + 29]
    mov rbx, r11
    and rbx, 511
    mov r12, r11
    shr r12, 9
    and r12, 511
    mov r13, r11
    shr r13, 18
    and r13, 511
    mov r11, r11
    shr r11, 27
    and r11, 511
    vmovq xmm7, qword ptr [r8 + rbx*8]
    vmovhps xmm7, xmm7, qword ptr [r8 + r12*8]
    vmovq xmm10, qword ptr [r8 + r13*8]
    vmovhps xmm10, xmm10, qword ptr [r8 + r11*8]
    vinserti128 ymm7, ymm7, xmm10, 1
    vpbroadcastd ymm10, dword ptr [rdi + 62]
    vpshufb ymm10, ymm10, ymm0
    vpand ymm10, ymm10, ymm1
    vpaddb ymm10, ymm10, ymm2
    vmovdqa ymm11, ymmword ptr [rdx + 384]
    vpsignb ymm11, ymm11, ymm10
    vpmaddubsw ymm11, ymm7, ymm11
    vmovdqa ymm13, ymmword ptr [rdx + 416]
    vpsignb ymm13, ymm13, ymm10
    vpmaddubsw ymm7, ymm7, ymm13
    # sums of group 5
    vpmaddwd ymm10, ymm12, ymm3
    vcvtdq2ps ymm10, ymm10
    vbroadcastss ymm12, dword ptr [rax + 40]
    vbroadcastss ymm13, dword ptr [rax + 104]
    vpblendd ymm12, ymm12, ymm13, 0xf0
    vfmadd231ps ymm4, ymm10, ymm12
    vpmaddwd ymm9, ymm9, ymm3
    vcvtdq2ps ymm9, ymm9
    vbroadcastss ymm10, dword ptr [rax + 44]
    vbroadcastss ymm12, dword ptr [rax + 108]
    vpblendd ymm10, ymm10, ymm12, 0xf0
    vfmadd231ps ymm5, ymm9, ymm10
    vmulps ymm8, ymm6, ymm8
    vmulps ymm8, ymm8, ymmword ptr [rip + {c6}]
    vshufps ymm6, ymm8, ymm8, 0x50
    vmulps ymm6, ymm6, ymmword ptr [rcx + 64]
    # products of group 7 of superblock +0
    mov r11, qword ptr [rdi + 33]
    shr r11, 4
    mov rbx, r11
    and rbx, 511
    mov r12, r11
    shr r12, 9
    and r12, 511
    mov r13, r11
    shr r13, 18
    and r13, 511
    mov r11, r11
    shr r11, 27
    and r11, 511
    vmovq xmm9, qword ptr [r8 + rbx*8]
    vmovhps xmm9, xmm9, qword ptr [r8 + r12*8]
    vmovq xmm10, qword ptr [r8 + r13*8]
    vmovhps xmm10, xmm10, qword ptr [r8 + r11*8]
    vinserti128 ymm9, ymm9, xmm10, 1
    vpbroadcastd ymm10, dword ptr [rdi + 66]
    vpshufb ymm10, ymm10, ymm0
    vpand ymm10, ymm10, ymm1
    vpaddb ymm10, ymm10, ymm2
    vmovdqa ymm12, ymmword ptr [rdx + 448]
    vpsignb ymm12, ymm12, ymm10
    vpmaddubsw ymm12, ymm9, ymm12
    vmovdqa ymm13, ymmword ptr [rdx + 480]
    vpsignb ymm13, ymm13, ymm10
    vpmaddubsw ymm9, ymm9, ymm13
    # sums of group 6
    vpmaddwd ymm10, ymm11, ymm3
    vcvtdq2ps ymm10, ymm10
    vbroadcastss ymm11, dword ptr [rax + 48]
    vbroadcastss ymm13, dword ptr [rax + 112]
    vpblendd ymm11, ymm11, ymm13, 0xf0
    vfmadd231ps ymm4, ymm10, ymm11
    vpmaddwd ymm7, ymm7, ymm3
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm10, dword ptr [rax + 52]
    vbroadcastss ymm11, dword ptr [rax + 116]
    vpblendd ymm10, ymm10, ymm11, 0xf0
    vfmadd231ps ymm5, ymm7, ymm10
    vmovaps ymmword ptr [r10 + 64], ymm6
    vshufps ymm6, ymm8, ymm8, 0xfa
    vmulps ymm6, ymm6, ymmword ptr [rcx + 96]
    vmovaps ymmword ptr [r10 + 96], ymm6
    # products of group 0 of superblock +1
    mov r11, qword ptr [rdi + 80]
    mov rbx, r11
    and rbx, 511
    mov r12, r11
    shr r12, 9
    and r12, 511
    mov r13, r11
    shr r13, 18
    and r13, 511
    mov r11, r11
    shr r11, 27
    and r11, 511
    vmovq xmm6, qword ptr [r8 + rbx*8]
    vmovhps xmm6, xmm6, qword ptr [r8 + r12*8]
    vmovq xmm7, qword ptr [r8 + r13*8]
    vmovhps xmm7, xmm7, qword ptr [r8 + r11*8]
    vinserti128 ymm6, ymm6, xmm7, 1
    vpbroadcastd ymm7, dword ptr [rdi + 116]
    vpshufb ymm7, ymm7, ymm0
    vpand ymm7, ymm7, ymm1
    vpaddb ymm7, ymm7, ymm2
    vmovdqa ymm8, ymmword ptr [rdx + 512]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm8, ymm6, ymm8
    vmovdqa ymm10, ymmword ptr [rdx + 544]
    vpsignb ymm10, ymm10, ymm7
    vpmaddubsw ymm6, ymm6, ymm10
    # sums of group 7
    vpmaddwd ymm7, ymm12, ymm3
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm10, dword ptr [rax + 56]
    vbroadcastss ymm11, dword ptr [rax + 120]
    vpblendd ymm10, ymm10, ymm11, 0xf0
    vfmadd231ps ymm4, ymm7, ymm10
    vpmaddwd ymm7, ymm9, ymm3
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm9, dword ptr [rax + 60]
    vbroadcastss ymm10, dword ptr [rax + 124]
    vpblendd ymm9, ymm9, ymm10, 0xf0
    vfmadd231ps ymm5, ymm7, ymm9
    # products of group 1 of superblock +1
    mov r11, qword ptr [rdi + 84]
    shr r11, 4
    mov rbx, r11
    and rbx, 511
    mov r12, r11
    shr r12, 9
    and r12, 511
    mov r13, r11
    shr r13, 18
    and r13, 511
    mov r11, r11
    shr r11, 27
    and r11, 511
    vmovq xmm7, qword ptr [r8 + rbx*8]
    vmovhps xmm7, xmm7, qword ptr [r8 + r12*8]
    vmovq xmm9, qword ptr [r8 + r13*8]
    vmovhps xmm9, xmm9, qword ptr [r8 + r11*8]
    vinserti128 ymm7, ymm7, xmm9, 1
    vpbroadcastd ymm9, dword ptr [rdi + 120]
    vpshufb ymm9, ymm9, ymm0
    vpand ymm9, ymm9, ymm1
    vpaddb ymm9, ymm9, ymm2
    vmovdqa ymm10, ymmword ptr [rdx + 576]
    vpsignb ymm10, ymm10, ymm9
    vpmaddubsw ymm10, ymm7, ymm10
    vmovdqa ymm11, ymmword ptr [rdx + 608]
    vpsignb ymm11, ymm11, ymm9
    vpmaddubsw ymm7, ymm7, ymm11
    add rdi, 78
    add rdx, 512
    add rcx, 64
    xor rax, 128
    xor r10, 128
    dec rsi
    jnz 2b
3:
    # sums of group 0
    vpmaddwd ymm8, ymm8, ymm3
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm9, dword ptr [rax]
    vbroadcastss ymm11, dword ptr [rax + 64]
    vpblendd ymm9, ymm9, ymm11, 0xf0
    vfmadd231ps ymm4, ymm8, ymm9
    vpmaddwd ymm6, ymm6, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm8, dword ptr [rax + 4]
    vbroadcastss ymm9, dword ptr [rax + 68]
    vpblendd ymm8, ymm8, ymm9, 0xf0
    vfmadd231ps ymm5, ymm6, ymm8
    # products of group 2 of superblock +0
    mov rcx, qword ptr [rdi + 11]
    mov rsi, rcx
    and rsi, 511
    mov r10, rcx
    shr r10, 9
    and r10, 511
    mov r11, rcx
    shr r11, 18
    and r11, 511
    mov rcx, rcx
    shr rcx, 27
    and rcx, 511
    vmovq xmm6, qword ptr [r8 + rsi*8]
    vmovhps xmm6, xmm6, qword ptr [r8 + r10*8]
    vmovq xmm8, qword ptr [r8 + r11*8]
    vmovhps xmm8, xmm8, qword ptr [r8 + rcx*8]
    vinserti128 ymm6, ymm6, xmm8, 1
    vpbroadcastd ymm8, dword ptr [rdi + 46]
    vpshufb ymm8, ymm8, ymm0
    vpand ymm8, ymm8, ymm1
    vpaddb ymm8, ymm8, ymm2
    vmovdqa ymm9, ymmword ptr [rdx + 128]
    vpsignb ymm9, ymm9, ymm8
    vpmaddubsw ymm12, ymm6, ymm9
    vmovdqa ymm9, ymmword ptr [rdx + 160]
    vpsignb ymm9, ymm9, ymm8
    vpmaddubsw ymm9, ymm6, ymm9
    # sums of group 1
    vpmaddwd ymm6, ymm10, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm8, dword ptr [rax + 8]
    vbroadcastss ymm10, dword ptr [rax + 72]
    vpblendd ymm8, ymm8, ymm10, 0xf0
    vfmadd231ps ymm4, ymm6, ymm8
    vpmaddwd ymm6, ymm7, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 12]
    vbroadcastss ymm8, dword ptr [rax + 76]
    vpblendd ymm7, ymm7, ymm8, 0xf0
    vfmadd231ps ymm5, ymm6, ymm7
    # products of group 3 of superblock +0
    mov rcx, qword ptr [rdi + 15]
    shr rcx, 4
    mov rsi, rcx
    and rsi, 511
    mov r10, rcx
    shr r10, 9
    and r10, 511
    mov r11, rcx
    shr r11, 18
    and r11, 511
    mov rcx, rcx
    shr rcx, 27
    and rcx, 511
    vmovq xmm6, qword ptr [r8 + rsi*8]
    vmovhps xmm6, xmm6, qword ptr [r8 + r10*8]
    vmovq xmm7, qword ptr [r8 + r11*8]
    vmovhps xmm7, xmm7, qword ptr [r8 + rcx*8]
    vinserti128 ymm6, ymm6, xmm7, 1
    vpbroadcastd ymm7, dword ptr [rdi + 50]
    vpshufb ymm7, ymm7, ymm0
    vpand ymm7, ymm7, ymm1
    vpaddb ymm7, ymm7, ymm2
    vmovdqa ymm8, ymmword ptr [rdx + 192]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm13, ymm6, ymm8
    vmovdqa ymm8, ymmword ptr [rdx + 224]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm10, ymm6, ymm8
    # sums of group 2
    vpmaddwd ymm6, ymm12, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 16]
    vbroadcastss ymm8, dword ptr [rax + 80]
    vpblendd ymm7, ymm7, ymm8, 0xf0
    vfmadd231ps ymm4, ymm6, ymm7
    vpmaddwd ymm6, ymm9, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 20]
    vbroadcastss ymm8, dword ptr [rax + 84]
    vpblendd ymm7, ymm7, ymm8, 0xf0
    vfmadd231ps ymm5, ymm6, ymm7
    # products of group 4 of superblock +0
    mov rcx, qword ptr [rdi + 20]
    mov rsi, rcx
    and rsi, 511
    mov r10, rcx
    shr r10, 9
    and r10, 511
    mov r11, rcx
    shr r11, 18
    and r11, 511
    mov rcx, rcx
    shr rcx, 27
    and rcx, 511
    vmovq xmm6, qword ptr [r8 + rsi*8]
    vmovhps xmm6, xmm6, qword ptr [r8 + r10*8]
    vmovq xmm7, qword ptr [r8 + r11*8]
    vmovhps xmm7, xmm7, qword ptr [r8 + rcx*8]
    vinserti128 ymm6, ymm6, xmm7, 1
    vpbroadcastd ymm7, dword ptr [rdi + 54]
    vpshufb ymm7, ymm7, ymm0
    vpand ymm7, ymm7, ymm1
    vpaddb ymm7, ymm7, ymm2
    vmovdqa ymm8, ymmword ptr [rdx + 256]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm14, ymm6, ymm8
    vmovdqa ymm8, ymmword ptr [rdx + 288]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm11, ymm6, ymm8
    # sums of group 3
    vpmaddwd ymm6, ymm13, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 24]
    vbroadcastss ymm8, dword ptr [rax + 88]
    vpblendd ymm7, ymm7, ymm8, 0xf0
    vfmadd231ps ymm4, ymm6, ymm7
    vpmaddwd ymm6, ymm10, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 28]
    vbroadcastss ymm8, dword ptr [rax + 92]
    vpblendd ymm7, ymm7, ymm8, 0xf0
    vfmadd231ps ymm5, ymm6, ymm7
    # products of group 5 of superblock +0
    mov rcx, qword ptr [rdi + 24]
    shr rcx, 4
    mov rsi, rcx
    and rsi, 511
    mov r10, rcx
    shr r10, 9
    and r10, 511
    mov r11, rcx
    shr r11, 18
    and r11, 511
    mov rcx, rcx
    shr rcx, 27
    and rcx, 511
    vmovq xmm6, qword ptr [r8 + rsi*8]
    vmovhps xmm6, xmm6, qword ptr [r8 + r10*8]
    vmovq xmm7, qword ptr [r8 + r11*8]
    vmovhps xmm7, xmm7, qword ptr [r8 + rcx*8]
    vinserti128 ymm6, ymm6, xmm7, 1
    vpbroadcastd ymm7, dword ptr [rdi + 58]
    vpshufb ymm7, ymm7, ymm0
    vpand ymm7, ymm7, ymm1
    vpaddb ymm7, ymm7, ymm2
    vmovdqa ymm8, ymmword ptr [rdx + 320]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm12, ymm6, ymm8
    vmovdqa ymm8, ymmword ptr [rdx + 352]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm9, ymm6, ymm8
    # sums of group 4
    vpmaddwd ymm6, ymm14, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 32]
    vbroadcastss ymm8, dword ptr [rax + 96]
    vpblendd ymm7, ymm7, ymm8, 0xf0
    vfmadd231ps ymm4, ymm6, ymm7
    vpmaddwd ymm6, ymm11, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 36]
    vbroadcastss ymm8, dword ptr [rax + 100]
    vpblendd ymm7, ymm7, ymm8, 0xf0
    vfmadd231ps ymm5, ymm6, ymm7
    # products of group 6 of superblock +0
    mov rcx, qword ptr [rdi + 29]
    mov rsi, rcx
    and rsi, 511
    mov r10, rcx
    shr r10, 9
    and r10, 511
    mov r11, rcx
    shr r11, 18
    and r11, 511
    mov rcx, rcx
    shr rcx, 27
    and rcx, 511
    vmovq xmm6, qword ptr [r8 + rsi*8]
    vmovhps xmm6, xmm6, qword ptr [r8 + r10*8]
    vmovq xmm7, qword ptr [r8 + r11*8]
    vmovhps xmm7, xmm7, qword ptr [r8 + rcx*8]
    vinserti128 ymm6, ymm6, xmm7, 1
    vpbroadcastd ymm7, dword ptr [rdi + 62]
    vpshufb ymm7, ymm7, ymm0
    vpand ymm7, ymm7, ymm1
    vpaddb ymm7, ymm7, ymm2
    vmovdqa ymm8, ymmword ptr [rdx + 384]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm11, ymm6, ymm8
    vmovdqa ymm8, ymmword ptr [rdx + 416]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm7, ymm6, ymm8
    # sums of group 5
    vpmaddwd ymm6, ymm12, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm8, dword ptr [rax + 40]
    vbroadcastss ymm10, dword ptr [rax + 104]
    vpblendd ymm8, ymm8, ymm10, 0xf0
    vfmadd231ps ymm4, ymm6, ymm8
    vpmaddwd ymm6, ymm9, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm8, dword ptr [rax + 44]
    vbroadcastss ymm9, dword ptr [rax + 108]
    vpblendd ymm8, ymm8, ymm9, 0xf0
    vfmadd231ps ymm5, ymm6, ymm8
    # products of group 7 of superblock +0
    mov rcx, qword ptr [rdi + 33]
    shr rcx, 4
    mov rsi, rcx
    and rsi, 511
    mov r10, rcx
    shr r10, 9
    and r10, 511
    mov r11, rcx
    shr r11, 18
    and r11, 511
    mov rcx, rcx
    shr rcx, 27
    and rcx, 511
    vmovq xmm6, qword ptr [r8 + rsi*8]
    vmovhps xmm6, xmm6, qword ptr [r8 + r10*8]
    vmovq xmm8, qword ptr [r8 + r11*8]
    vmovhps xmm8, xmm8, qword ptr [r8 + rcx*8]
    vinserti128 ymm6, ymm6, xmm8, 1
    vpbroadcastd ymm8, dword ptr [rdi + 66]
    vpshufb ymm8, ymm8, ymm0
    vpand ymm8, ymm8, ymm1
    vpaddb ymm8, ymm8, ymm2
    vmovdqa ymm0, ymmword ptr [rdx + 448]
    vpsignb ymm0, ymm0, ymm8
    vpmaddubsw ymm12, ymm6, ymm0
    vmovdqa ymm0, ymmword ptr [rdx + 480]
    vpsignb ymm0, ymm0, ymm8
    vpmaddubsw ymm9, ymm6, ymm0
    # sums of group 6
    vpmaddwd ymm0, ymm11, ymm3
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm1, dword ptr [rax + 48]
    vbroadcastss ymm2, dword ptr [rax + 112]
    vpblendd ymm1, ymm1, ymm2, 0xf0
    vfmadd231ps ymm4, ymm0, ymm1
    vpmaddwd ymm0, ymm7, ymm3
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm1, dword ptr [rax + 52]
    vbroadcastss ymm2, dword ptr [rax + 116]
    vpblendd ymm1, ymm1, ymm2, 0xf0
    vfmadd231ps ymm5, ymm0, ymm1
    # sums of group 7
    vpmaddwd ymm0, ymm12, ymm3
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm1, dword ptr [rax + 56]
    vbroadcastss ymm2, dword ptr [rax + 120]
    vpblendd ymm1, ymm1, ymm2, 0xf0
    vfmadd231ps ymm4, ymm0, ymm1
    vpmaddwd ymm0, ymm9, ymm3
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm1, dword ptr [rax + 60]
    vbroadcastss ymm2, dword ptr [rax + 124]
    vpblendd ymm1, ymm1, ymm2, 0xf0
    vfmadd231ps ymm5, ymm0, ymm1
    # the lanes summed
    vxorps xmm0, xmm0, xmm0
    vaddps ymm4, ymm4, ymm0
    vextractf128 xmm0, ymm4, 1
    vaddps xmm0, xmm4, xmm0
    vmovhlps xmm1, xmm0, xmm0
    vaddps xmm0, xmm0, xmm1
    vmovshdup xmm1, xmm0
    vaddss xmm0, xmm0, xmm1
    vmovss dword ptr [r9], xmm0
    # the lanes summed
    vxorps xmm0, xmm0, xmm0
    vaddps ymm5, ymm5, ymm0
    vextractf128 xmm0, ymm5, 1
    vaddps xmm0, xmm5, xmm0
    vmovhlps xmm1, xmm0, xmm0
    vaddps xmm0, xmm0, xmm1
    vmovshdup xmm1, xmm0
    vaddss xmm0, xmm0, xmm1
    vmovss dword ptr [r9 + 4], xmm0
    mov rsp, rbp
    pop rbp
    pop r13
    pop r12
    pop rbx
    vzeroupper
    ret
4:
    vxorps xmm0, xmm0, xmm0
    vmovss dword ptr [r9], xmm0
    vmovss dword ptr [r9 + 4], xmm0
    mov rsp, rbp
    pop rbp
    pop r13
    pop r12
    pop rbx
    vzeroupper
    ret
# c0, aligned to 32:
#   00 00 00 00 00 00 00 00 01 01 01 01 01 01 01 01
#   02 02 02 02 02 02 02 02 03 03 03 03 03 03 03 03
# c1, aligned to 8:
#   01 02 04 08 10 20 40 80
# c2, aligned to 4:
#   7f 7f 7f 7f
# c3, aligned to 4:
#   01 00 01 00
# c4, aligned to 32:
#   0f 00 00 00 0f 00 00 00 0f 00 00 00 0f 00 00 00
#   0f 00 00 00 0f 00 00 00 0f 00 00 00 0f 00 00 00
# c5, aligned to 32:
#   01 00 00 00 01 00 00 00 01 00 00 00 01 00 00 00
#   01 00 00 00 01 00 00 00 01 00 00 00 01 00 00 00
# c6, aligned to 32:
#   00 00 00 3e 00 00 00 3e 00 00 00 3e 00 00 00 3e
#   00 00 00 3e 00 00 00 3e 00 00 00 3e 00 00 00 3e
