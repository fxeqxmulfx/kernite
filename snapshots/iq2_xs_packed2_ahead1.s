# iq2_xs_packed2_ahead1(gpr, gpr, gpr, gpr, gpr, gpr) -> ()
# frame: 256 bytes aligned to 256, saves [rbx, r12, r13]
# row=rdi sbs=rsi quants=rdx scales=rcx grid=r8 out=r9
# spread=ymm0 bit=ymm1 max8=ymm2 one16=ymm3 acc0=ymm4 acc1=ymm5
# cur=rax nxt=r10 p0a=ymm8 p0b=ymm6 p1a=ymm10 p1b=ymm8
# p2a=ymm11 p2b=ymm9 p3a=ymm12 p3b=ymm10 p4a=ymm11 p4b=ymm9
# p5a=ymm10 p5b=ymm8 p6a=ymm10 p6b=ymm8 p7a=ymm8 p7b=ymm6
# d=ymm6 steps=ymm7 low_scales=ymm8 high_scales=ymm7 pair_scale=ymm9,ymm8,ymm6 index0=rbx,rsi
# index1=r12,r10 index2=r13,r11 index3=r11,rcx packed_indices=r11,rcx levels=ymm6,ymm8,ymm9,ymm10 rows=ymm7,ymm9,ymm10,ymm11
# signs=ymm7,ymm9,ymm10,ymm11 x=ymm8,ymm9,ymm10,ymm11,ymm12,ymm13,ymm0 lanes=ymm7,ymm6,ymm9,ymm8,ymm10,ymm11,ymm0 low_scale=ymm8,ymm7,ymm10,ymm9,ymm11,ymm12,ymm1 high_scale=ymm9,ymm8,ymm11,ymm10,ymm12,ymm13,ymm2 scale=ymm8,ymm7,ymm10,ymm9,ymm11,ymm12,ymm1
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
    dec rsi
    jz 3f
2:
    # sums of group 0
    vpmaddwd ymm7, ymm8, ymm3
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rax]
    vbroadcastss ymm9, dword ptr [rax + 64]
    vpblendd ymm8, ymm8, ymm9, 0xf0
    vfmadd231ps ymm4, ymm7, ymm8
    vpmaddwd ymm6, ymm6, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 4]
    vbroadcastss ymm8, dword ptr [rax + 68]
    vpblendd ymm7, ymm7, ymm8, 0xf0
    vfmadd231ps ymm5, ymm6, ymm7
    # pair scales of superblock +1
    vpbroadcastw xmm6, word ptr [rdi + 78]
    vcvtph2ps ymm6, xmm6
    vpmovzxbd ymm7, qword ptr [rdi + 148]
    vpermq ymm7, ymm7, 0xd8
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
    vmovq xmm8, qword ptr [r8 + rbx*8]
    vmovhps xmm8, xmm8, qword ptr [r8 + r12*8]
    vmovq xmm9, qword ptr [r8 + r13*8]
    vmovhps xmm9, xmm9, qword ptr [r8 + r11*8]
    vinserti128 ymm8, ymm8, xmm9, 1
    vpbroadcastd ymm9, dword ptr [rdi + 42]
    vpshufb ymm9, ymm9, ymm0
    vpand ymm9, ymm9, ymm1
    vpaddb ymm9, ymm9, ymm2
    vmovdqa ymm10, ymmword ptr [rdx + 64]
    vpsignb ymm10, ymm10, ymm9
    vpmaddubsw ymm10, ymm8, ymm10
    vmovdqa ymm11, ymmword ptr [rdx + 96]
    vpsignb ymm11, ymm11, ymm9
    vpmaddubsw ymm8, ymm8, ymm11
    # sums of group 1
    vpmaddwd ymm9, ymm10, ymm3
    vcvtdq2ps ymm9, ymm9
    vbroadcastss ymm10, dword ptr [rax + 8]
    vbroadcastss ymm11, dword ptr [rax + 72]
    vpblendd ymm10, ymm10, ymm11, 0xf0
    vfmadd231ps ymm4, ymm9, ymm10
    vpmaddwd ymm8, ymm8, ymm3
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm9, dword ptr [rax + 12]
    vbroadcastss ymm10, dword ptr [rax + 76]
    vpblendd ymm9, ymm9, ymm10, 0xf0
    vfmadd231ps ymm5, ymm8, ymm9
    vpand ymm8, ymm7, ymmword ptr [rip + {c4}]
    vpsrld ymm7, ymm7, 4
    vpaddd ymm8, ymm8, ymm8
    vpor ymm8, ymm8, ymmword ptr [rip + {c5}]
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
    vmovq xmm10, qword ptr [r8 + r13*8]
    vmovhps xmm10, xmm10, qword ptr [r8 + r11*8]
    vinserti128 ymm9, ymm9, xmm10, 1
    vpbroadcastd ymm10, dword ptr [rdi + 46]
    vpshufb ymm10, ymm10, ymm0
    vpand ymm10, ymm10, ymm1
    vpaddb ymm10, ymm10, ymm2
    vmovdqa ymm11, ymmword ptr [rdx + 128]
    vpsignb ymm11, ymm11, ymm10
    vpmaddubsw ymm11, ymm9, ymm11
    vmovdqa ymm12, ymmword ptr [rdx + 160]
    vpsignb ymm12, ymm12, ymm10
    vpmaddubsw ymm9, ymm9, ymm12
    # sums of group 2
    vpmaddwd ymm10, ymm11, ymm3
    vcvtdq2ps ymm10, ymm10
    vbroadcastss ymm11, dword ptr [rax + 16]
    vbroadcastss ymm12, dword ptr [rax + 80]
    vpblendd ymm11, ymm11, ymm12, 0xf0
    vfmadd231ps ymm4, ymm10, ymm11
    vpmaddwd ymm9, ymm9, ymm3
    vcvtdq2ps ymm9, ymm9
    vbroadcastss ymm10, dword ptr [rax + 20]
    vbroadcastss ymm11, dword ptr [rax + 84]
    vpblendd ymm10, ymm10, ymm11, 0xf0
    vfmadd231ps ymm5, ymm9, ymm10
    vcvtdq2ps ymm8, ymm8
    vmulps ymm8, ymm6, ymm8
    vmulps ymm8, ymm8, ymmword ptr [rip + {c6}]
    vshufps ymm9, ymm8, ymm8, 0x50
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
    vmovdqa ymm12, ymmword ptr [rdx + 192]
    vpsignb ymm12, ymm12, ymm11
    vpmaddubsw ymm12, ymm10, ymm12
    vmovdqa ymm13, ymmword ptr [rdx + 224]
    vpsignb ymm13, ymm13, ymm11
    vpmaddubsw ymm10, ymm10, ymm13
    # sums of group 3
    vpmaddwd ymm11, ymm12, ymm3
    vcvtdq2ps ymm11, ymm11
    vbroadcastss ymm12, dword ptr [rax + 24]
    vbroadcastss ymm13, dword ptr [rax + 88]
    vpblendd ymm12, ymm12, ymm13, 0xf0
    vfmadd231ps ymm4, ymm11, ymm12
    vpmaddwd ymm10, ymm10, ymm3
    vcvtdq2ps ymm10, ymm10
    vbroadcastss ymm11, dword ptr [rax + 28]
    vbroadcastss ymm12, dword ptr [rax + 92]
    vpblendd ymm11, ymm11, ymm12, 0xf0
    vfmadd231ps ymm5, ymm10, ymm11
    vmulps ymm9, ymm9, ymmword ptr [rcx + 64]
    vmovaps ymmword ptr [r10], ymm9
    vshufps ymm8, ymm8, ymm8, 0xfa
    vmulps ymm8, ymm8, ymmword ptr [rcx + 96]
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
    vmovq xmm9, qword ptr [r8 + rbx*8]
    vmovhps xmm9, xmm9, qword ptr [r8 + r12*8]
    vmovq xmm10, qword ptr [r8 + r13*8]
    vmovhps xmm10, xmm10, qword ptr [r8 + r11*8]
    vinserti128 ymm9, ymm9, xmm10, 1
    vpbroadcastd ymm10, dword ptr [rdi + 54]
    vpshufb ymm10, ymm10, ymm0
    vpand ymm10, ymm10, ymm1
    vpaddb ymm10, ymm10, ymm2
    vmovdqa ymm11, ymmword ptr [rdx + 256]
    vpsignb ymm11, ymm11, ymm10
    vpmaddubsw ymm11, ymm9, ymm11
    vmovdqa ymm12, ymmword ptr [rdx + 288]
    vpsignb ymm12, ymm12, ymm10
    vpmaddubsw ymm9, ymm9, ymm12
    # sums of group 4
    vpmaddwd ymm10, ymm11, ymm3
    vcvtdq2ps ymm10, ymm10
    vbroadcastss ymm11, dword ptr [rax + 32]
    vbroadcastss ymm12, dword ptr [rax + 96]
    vpblendd ymm11, ymm11, ymm12, 0xf0
    vfmadd231ps ymm4, ymm10, ymm11
    vpmaddwd ymm9, ymm9, ymm3
    vcvtdq2ps ymm9, ymm9
    vbroadcastss ymm10, dword ptr [rax + 36]
    vbroadcastss ymm11, dword ptr [rax + 100]
    vpblendd ymm10, ymm10, ymm11, 0xf0
    vfmadd231ps ymm5, ymm9, ymm10
    vmovaps ymmword ptr [r10 + 32], ymm8
    vpaddd ymm7, ymm7, ymm7
    vpor ymm7, ymm7, ymmword ptr [rip + {c5}]
    vcvtdq2ps ymm7, ymm7
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
    vmovq xmm8, qword ptr [r8 + rbx*8]
    vmovhps xmm8, xmm8, qword ptr [r8 + r12*8]
    vmovq xmm9, qword ptr [r8 + r13*8]
    vmovhps xmm9, xmm9, qword ptr [r8 + r11*8]
    vinserti128 ymm8, ymm8, xmm9, 1
    vpbroadcastd ymm9, dword ptr [rdi + 58]
    vpshufb ymm9, ymm9, ymm0
    vpand ymm9, ymm9, ymm1
    vpaddb ymm9, ymm9, ymm2
    vmovdqa ymm10, ymmword ptr [rdx + 320]
    vpsignb ymm10, ymm10, ymm9
    vpmaddubsw ymm10, ymm8, ymm10
    vmovdqa ymm11, ymmword ptr [rdx + 352]
    vpsignb ymm11, ymm11, ymm9
    vpmaddubsw ymm8, ymm8, ymm11
    # sums of group 5
    vpmaddwd ymm9, ymm10, ymm3
    vcvtdq2ps ymm9, ymm9
    vbroadcastss ymm10, dword ptr [rax + 40]
    vbroadcastss ymm11, dword ptr [rax + 104]
    vpblendd ymm10, ymm10, ymm11, 0xf0
    vfmadd231ps ymm4, ymm9, ymm10
    vpmaddwd ymm8, ymm8, ymm3
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm9, dword ptr [rax + 44]
    vbroadcastss ymm10, dword ptr [rax + 108]
    vpblendd ymm9, ymm9, ymm10, 0xf0
    vfmadd231ps ymm5, ymm8, ymm9
    vmulps ymm7, ymm6, ymm7
    vmulps ymm7, ymm7, ymmword ptr [rip + {c6}]
    vshufps ymm6, ymm7, ymm7, 0x50
    vmulps ymm6, ymm6, ymmword ptr [rcx + 64]
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
    vmovq xmm8, qword ptr [r8 + rbx*8]
    vmovhps xmm8, xmm8, qword ptr [r8 + r12*8]
    vmovq xmm9, qword ptr [r8 + r13*8]
    vmovhps xmm9, xmm9, qword ptr [r8 + r11*8]
    vinserti128 ymm8, ymm8, xmm9, 1
    vpbroadcastd ymm9, dword ptr [rdi + 62]
    vpshufb ymm9, ymm9, ymm0
    vpand ymm9, ymm9, ymm1
    vpaddb ymm9, ymm9, ymm2
    vmovdqa ymm10, ymmword ptr [rdx + 384]
    vpsignb ymm10, ymm10, ymm9
    vpmaddubsw ymm10, ymm8, ymm10
    vmovdqa ymm11, ymmword ptr [rdx + 416]
    vpsignb ymm11, ymm11, ymm9
    vpmaddubsw ymm8, ymm8, ymm11
    # sums of group 6
    vpmaddwd ymm9, ymm10, ymm3
    vcvtdq2ps ymm9, ymm9
    vbroadcastss ymm10, dword ptr [rax + 48]
    vbroadcastss ymm11, dword ptr [rax + 112]
    vpblendd ymm10, ymm10, ymm11, 0xf0
    vfmadd231ps ymm4, ymm9, ymm10
    vpmaddwd ymm8, ymm8, ymm3
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm9, dword ptr [rax + 52]
    vbroadcastss ymm10, dword ptr [rax + 116]
    vpblendd ymm9, ymm9, ymm10, 0xf0
    vfmadd231ps ymm5, ymm8, ymm9
    vmovaps ymmword ptr [r10 + 64], ymm6
    vshufps ymm6, ymm7, ymm7, 0xfa
    vmulps ymm6, ymm6, ymmword ptr [rcx + 96]
    vmovaps ymmword ptr [r10 + 96], ymm6
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
    vmovq xmm6, qword ptr [r8 + rbx*8]
    vmovhps xmm6, xmm6, qword ptr [r8 + r12*8]
    vmovq xmm7, qword ptr [r8 + r13*8]
    vmovhps xmm7, xmm7, qword ptr [r8 + r11*8]
    vinserti128 ymm6, ymm6, xmm7, 1
    vpbroadcastd ymm7, dword ptr [rdi + 66]
    vpshufb ymm7, ymm7, ymm0
    vpand ymm7, ymm7, ymm1
    vpaddb ymm7, ymm7, ymm2
    vmovdqa ymm8, ymmword ptr [rdx + 448]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm8, ymm6, ymm8
    vmovdqa ymm9, ymmword ptr [rdx + 480]
    vpsignb ymm9, ymm9, ymm7
    vpmaddubsw ymm6, ymm6, ymm9
    # sums of group 7
    vpmaddwd ymm7, ymm8, ymm3
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rax + 56]
    vbroadcastss ymm9, dword ptr [rax + 120]
    vpblendd ymm8, ymm8, ymm9, 0xf0
    vfmadd231ps ymm4, ymm7, ymm8
    vpmaddwd ymm6, ymm6, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 60]
    vbroadcastss ymm8, dword ptr [rax + 124]
    vpblendd ymm7, ymm7, ymm8, 0xf0
    vfmadd231ps ymm5, ymm6, ymm7
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
    vmovdqa ymm9, ymmword ptr [rdx + 544]
    vpsignb ymm9, ymm9, ymm7
    vpmaddubsw ymm6, ymm6, ymm9
    add rdi, 78
    add rdx, 512
    add rcx, 64
    xor rax, 128
    xor r10, 128
    dec rsi
    jnz 2b
3:
    # sums of group 0
    vpmaddwd ymm7, ymm8, ymm3
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rax]
    vbroadcastss ymm9, dword ptr [rax + 64]
    vpblendd ymm8, ymm8, ymm9, 0xf0
    vfmadd231ps ymm4, ymm7, ymm8
    vpmaddwd ymm6, ymm6, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 4]
    vbroadcastss ymm8, dword ptr [rax + 68]
    vpblendd ymm7, ymm7, ymm8, 0xf0
    vfmadd231ps ymm5, ymm6, ymm7
    # products of group 1 of superblock +0
    mov rcx, qword ptr [rdi + 6]
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
    vpbroadcastd ymm7, dword ptr [rdi + 42]
    vpshufb ymm7, ymm7, ymm0
    vpand ymm7, ymm7, ymm1
    vpaddb ymm7, ymm7, ymm2
    vmovdqa ymm8, ymmword ptr [rdx + 64]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm10, ymm6, ymm8
    vmovdqa ymm8, ymmword ptr [rdx + 96]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm8, ymm6, ymm8
    # sums of group 1
    vpmaddwd ymm6, ymm10, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 8]
    vbroadcastss ymm9, dword ptr [rax + 72]
    vpblendd ymm7, ymm7, ymm9, 0xf0
    vfmadd231ps ymm4, ymm6, ymm7
    vpmaddwd ymm6, ymm8, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 12]
    vbroadcastss ymm8, dword ptr [rax + 76]
    vpblendd ymm7, ymm7, ymm8, 0xf0
    vfmadd231ps ymm5, ymm6, ymm7
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
    vmovq xmm7, qword ptr [r8 + r11*8]
    vmovhps xmm7, xmm7, qword ptr [r8 + rcx*8]
    vinserti128 ymm6, ymm6, xmm7, 1
    vpbroadcastd ymm7, dword ptr [rdi + 46]
    vpshufb ymm7, ymm7, ymm0
    vpand ymm7, ymm7, ymm1
    vpaddb ymm7, ymm7, ymm2
    vmovdqa ymm8, ymmword ptr [rdx + 128]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm11, ymm6, ymm8
    vmovdqa ymm8, ymmword ptr [rdx + 160]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm9, ymm6, ymm8
    # sums of group 2
    vpmaddwd ymm6, ymm11, ymm3
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
    vpmaddubsw ymm12, ymm6, ymm8
    vmovdqa ymm8, ymmword ptr [rdx + 224]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm10, ymm6, ymm8
    # sums of group 3
    vpmaddwd ymm6, ymm12, ymm3
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
    vpmaddubsw ymm11, ymm6, ymm8
    vmovdqa ymm8, ymmword ptr [rdx + 288]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm9, ymm6, ymm8
    # sums of group 4
    vpmaddwd ymm6, ymm11, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 32]
    vbroadcastss ymm8, dword ptr [rax + 96]
    vpblendd ymm7, ymm7, ymm8, 0xf0
    vfmadd231ps ymm4, ymm6, ymm7
    vpmaddwd ymm6, ymm9, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 36]
    vbroadcastss ymm8, dword ptr [rax + 100]
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
    vpmaddubsw ymm10, ymm6, ymm8
    vmovdqa ymm8, ymmword ptr [rdx + 352]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm8, ymm6, ymm8
    # sums of group 5
    vpmaddwd ymm6, ymm10, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 40]
    vbroadcastss ymm9, dword ptr [rax + 104]
    vpblendd ymm7, ymm7, ymm9, 0xf0
    vfmadd231ps ymm4, ymm6, ymm7
    vpmaddwd ymm6, ymm8, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 44]
    vbroadcastss ymm8, dword ptr [rax + 108]
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
    vpmaddubsw ymm10, ymm6, ymm8
    vmovdqa ymm8, ymmword ptr [rdx + 416]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm8, ymm6, ymm8
    # sums of group 6
    vpmaddwd ymm6, ymm10, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 48]
    vbroadcastss ymm9, dword ptr [rax + 112]
    vpblendd ymm7, ymm7, ymm9, 0xf0
    vfmadd231ps ymm4, ymm6, ymm7
    vpmaddwd ymm6, ymm8, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 52]
    vbroadcastss ymm8, dword ptr [rax + 116]
    vpblendd ymm7, ymm7, ymm8, 0xf0
    vfmadd231ps ymm5, ymm6, ymm7
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
    vmovq xmm7, qword ptr [r8 + r11*8]
    vmovhps xmm7, xmm7, qword ptr [r8 + rcx*8]
    vinserti128 ymm6, ymm6, xmm7, 1
    vpbroadcastd ymm7, dword ptr [rdi + 66]
    vpshufb ymm7, ymm7, ymm0
    vpand ymm7, ymm7, ymm1
    vpaddb ymm7, ymm7, ymm2
    vmovdqa ymm0, ymmword ptr [rdx + 448]
    vpsignb ymm0, ymm0, ymm7
    vpmaddubsw ymm8, ymm6, ymm0
    vmovdqa ymm0, ymmword ptr [rdx + 480]
    vpsignb ymm0, ymm0, ymm7
    vpmaddubsw ymm6, ymm6, ymm0
    # sums of group 7
    vpmaddwd ymm0, ymm8, ymm3
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm1, dword ptr [rax + 56]
    vbroadcastss ymm2, dword ptr [rax + 120]
    vpblendd ymm1, ymm1, ymm2, 0xf0
    vfmadd231ps ymm4, ymm0, ymm1
    vpmaddwd ymm0, ymm6, ymm3
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
