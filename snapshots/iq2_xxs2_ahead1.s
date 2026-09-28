# iq2_xxs2_ahead1(gpr, gpr, gpr, gpr, gpr, gpr) -> ()
# frame: 256 bytes aligned to 256, saves [rbx, r12, r13]
# row=rdi sbs=rsi quants=rdx scales=rcx grid=r8 out=r9
# spread=ymm0 bit=ymm1 max8=ymm2 one16=ymm3 acc0=ymm4 acc1=ymm5
# cur=rax nxt=r10 p0a=ymm8 p0b=ymm6 p1a=ymm10 p1b=ymm8
# p2a=ymm11 p2b=ymm9 p3a=ymm11 p3b=ymm9 p4a=ymm11 p4b=ymm8
# p5a=ymm11 p5b=ymm8 p6a=ymm9 p6b=ymm7 p7a=ymm9 p7b=ymm7
# steps=ymm6 words=ymm6 fields=ymm7 field=ymm8 parity=ymm9 d=ymm7
# scale=ymm6 pair_scale=ymm7,ymm6 index0=r11,rcx index1=rbx,rsi index2=r12,r10 index3=r13,r11,rdi
# levels=ymm6,ymm8,ymm9,ymm7 rows=ymm7,ymm9,ymm10,ymm8 signs=ymm7,ymm9,ymm10,ymm8 x=ymm8,ymm9,ymm10,ymm11,ymm12,ymm0 lanes=ymm7,ymm6,ymm9,ymm8,ymm10,ymm0 group_scale=ymm8,ymm7,ymm10,ymm9,ymm11,ymm1
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
    # signs of superblock +0
    vmovdqu ymm6, ymmword ptr [rdi + 2]
    vshufps ymm6, ymm6, ymmword ptr [rdi + 34], 0xdd
    vpermq ymm6, ymm6, 0xd8
    vpand ymm7, ymm6, ymmword ptr [rip + {c4}]
    vpaddd ymm8, ymm6, ymm6
    vpand ymm8, ymm8, ymmword ptr [rip + {c5}]
    vpor ymm7, ymm7, ymm8
    vpslld ymm8, ymm6, 2
    vpand ymm8, ymm8, ymmword ptr [rip + {c6}]
    vpor ymm7, ymm7, ymm8
    vpslld ymm8, ymm6, 3
    vpand ymm8, ymm8, ymmword ptr [rip + {c7}]
    vpor ymm7, ymm7, ymm8
    vpsrld ymm6, ymm6, 27
    vpaddw ymm9, ymm7, ymm7
    vpxor ymm9, ymm9, ymm7
    vpsllw ymm8, ymm9, 2
    vpxor ymm9, ymm9, ymm8
    vpsllw ymm8, ymm9, 4
    vpxor ymm9, ymm9, ymm8
    vpand ymm9, ymm9, ymmword ptr [rip + {c8}]
    vpor ymm7, ymm7, ymm9
    vmovdqa ymmword ptr [rax + 64], ymm7
    # pair scales of superblock +0
    vpbroadcastw xmm7, word ptr [rdi]
    vcvtph2ps ymm7, xmm7
    vpor ymm6, ymm6, ymmword ptr [rip + {c9}]
    vcvtdq2ps ymm6, ymm6
    vmulps ymm6, ymm7, ymm6
    vmulps ymm6, ymm6, ymmword ptr [rip + {c10}]
    vpermq ymm6, ymm6, 0xd8
    vshufps ymm7, ymm6, ymm6, 0x50
    vmulps ymm7, ymm7, ymmword ptr [rcx]
    vmovaps ymmword ptr [rax], ymm7
    vshufps ymm6, ymm6, ymm6, 0xfa
    vmulps ymm6, ymm6, ymmword ptr [rcx + 32]
    vmovaps ymmword ptr [rax + 32], ymm6
    # products of group 0 of superblock +0
    movzx r11d, byte ptr [rdi + 2]
    movzx ebx, byte ptr [rdi + 3]
    movzx r12d, byte ptr [rdi + 4]
    movzx r13d, byte ptr [rdi + 5]
    vmovq xmm6, qword ptr [r8 + r11*8]
    vmovhps xmm6, xmm6, qword ptr [r8 + rbx*8]
    vmovq xmm7, qword ptr [r8 + r12*8]
    vmovhps xmm7, xmm7, qword ptr [r8 + r13*8]
    vinserti128 ymm6, ymm6, xmm7, 1
    vpbroadcastd ymm7, dword ptr [rax + 64]
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
    vfmadd231ps ymm4, ymm7, ymm8
    vpmaddwd ymm6, ymm6, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 4]
    vfmadd231ps ymm5, ymm6, ymm7
    # signs of superblock +1
    vmovdqu ymm6, ymmword ptr [rdi + 68]
    vshufps ymm6, ymm6, ymmword ptr [rdi + 100], 0xdd
    vpermq ymm6, ymm6, 0xd8
    vpand ymm7, ymm6, ymmword ptr [rip + {c4}]
    # products of group 1 of superblock +0
    movzx r11d, byte ptr [rdi + 10]
    movzx ebx, byte ptr [rdi + 11]
    movzx r12d, byte ptr [rdi + 12]
    movzx r13d, byte ptr [rdi + 13]
    vmovq xmm8, qword ptr [r8 + r11*8]
    vmovhps xmm8, xmm8, qword ptr [r8 + rbx*8]
    vmovq xmm9, qword ptr [r8 + r12*8]
    vmovhps xmm9, xmm9, qword ptr [r8 + r13*8]
    vinserti128 ymm8, ymm8, xmm9, 1
    vpbroadcastd ymm9, dword ptr [rax + 68]
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
    vfmadd231ps ymm4, ymm9, ymm10
    vpmaddwd ymm8, ymm8, ymm3
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm9, dword ptr [rax + 12]
    vfmadd231ps ymm5, ymm8, ymm9
    vpaddd ymm8, ymm6, ymm6
    vpand ymm8, ymm8, ymmword ptr [rip + {c5}]
    vpor ymm7, ymm7, ymm8
    vpslld ymm8, ymm6, 2
    # products of group 2 of superblock +0
    movzx r11d, byte ptr [rdi + 18]
    movzx ebx, byte ptr [rdi + 19]
    movzx r12d, byte ptr [rdi + 20]
    movzx r13d, byte ptr [rdi + 21]
    vmovq xmm9, qword ptr [r8 + r11*8]
    vmovhps xmm9, xmm9, qword ptr [r8 + rbx*8]
    vmovq xmm10, qword ptr [r8 + r12*8]
    vmovhps xmm10, xmm10, qword ptr [r8 + r13*8]
    vinserti128 ymm9, ymm9, xmm10, 1
    vpbroadcastd ymm10, dword ptr [rax + 72]
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
    vfmadd231ps ymm4, ymm10, ymm11
    vpmaddwd ymm9, ymm9, ymm3
    vcvtdq2ps ymm9, ymm9
    vbroadcastss ymm10, dword ptr [rax + 20]
    vfmadd231ps ymm5, ymm9, ymm10
    vpand ymm8, ymm8, ymmword ptr [rip + {c6}]
    vpor ymm7, ymm7, ymm8
    vpslld ymm8, ymm6, 3
    vpand ymm8, ymm8, ymmword ptr [rip + {c7}]
    # products of group 3 of superblock +0
    movzx r11d, byte ptr [rdi + 26]
    movzx ebx, byte ptr [rdi + 27]
    movzx r12d, byte ptr [rdi + 28]
    movzx r13d, byte ptr [rdi + 29]
    vmovq xmm9, qword ptr [r8 + r11*8]
    vmovhps xmm9, xmm9, qword ptr [r8 + rbx*8]
    vmovq xmm10, qword ptr [r8 + r12*8]
    vmovhps xmm10, xmm10, qword ptr [r8 + r13*8]
    vinserti128 ymm9, ymm9, xmm10, 1
    vpbroadcastd ymm10, dword ptr [rax + 76]
    vpshufb ymm10, ymm10, ymm0
    vpand ymm10, ymm10, ymm1
    vpaddb ymm10, ymm10, ymm2
    vmovdqa ymm11, ymmword ptr [rdx + 192]
    vpsignb ymm11, ymm11, ymm10
    vpmaddubsw ymm11, ymm9, ymm11
    vmovdqa ymm12, ymmword ptr [rdx + 224]
    vpsignb ymm12, ymm12, ymm10
    vpmaddubsw ymm9, ymm9, ymm12
    # sums of group 3
    vpmaddwd ymm10, ymm11, ymm3
    vcvtdq2ps ymm10, ymm10
    vbroadcastss ymm11, dword ptr [rax + 24]
    vfmadd231ps ymm4, ymm10, ymm11
    vpmaddwd ymm9, ymm9, ymm3
    vcvtdq2ps ymm9, ymm9
    vbroadcastss ymm10, dword ptr [rax + 28]
    vfmadd231ps ymm5, ymm9, ymm10
    vpor ymm7, ymm7, ymm8
    vpsrld ymm6, ymm6, 27
    vpaddw ymm9, ymm7, ymm7
    vpxor ymm9, ymm9, ymm7
    # products of group 4 of superblock +0
    movzx r11d, byte ptr [rdi + 34]
    movzx ebx, byte ptr [rdi + 35]
    movzx r12d, byte ptr [rdi + 36]
    movzx r13d, byte ptr [rdi + 37]
    vmovq xmm8, qword ptr [r8 + r11*8]
    vmovhps xmm8, xmm8, qword ptr [r8 + rbx*8]
    vmovq xmm10, qword ptr [r8 + r12*8]
    vmovhps xmm10, xmm10, qword ptr [r8 + r13*8]
    vinserti128 ymm8, ymm8, xmm10, 1
    vpbroadcastd ymm10, dword ptr [rax + 80]
    vpshufb ymm10, ymm10, ymm0
    vpand ymm10, ymm10, ymm1
    vpaddb ymm10, ymm10, ymm2
    vmovdqa ymm11, ymmword ptr [rdx + 256]
    vpsignb ymm11, ymm11, ymm10
    vpmaddubsw ymm11, ymm8, ymm11
    vmovdqa ymm12, ymmword ptr [rdx + 288]
    vpsignb ymm12, ymm12, ymm10
    vpmaddubsw ymm8, ymm8, ymm12
    # sums of group 4
    vpmaddwd ymm10, ymm11, ymm3
    vcvtdq2ps ymm10, ymm10
    vbroadcastss ymm11, dword ptr [rax + 32]
    vfmadd231ps ymm4, ymm10, ymm11
    vpmaddwd ymm8, ymm8, ymm3
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm10, dword ptr [rax + 36]
    vfmadd231ps ymm5, ymm8, ymm10
    vpsllw ymm8, ymm9, 2
    vpxor ymm9, ymm9, ymm8
    vpsllw ymm8, ymm9, 4
    vpxor ymm9, ymm9, ymm8
    # products of group 5 of superblock +0
    movzx r11d, byte ptr [rdi + 42]
    movzx ebx, byte ptr [rdi + 43]
    movzx r12d, byte ptr [rdi + 44]
    movzx r13d, byte ptr [rdi + 45]
    vmovq xmm8, qword ptr [r8 + r11*8]
    vmovhps xmm8, xmm8, qword ptr [r8 + rbx*8]
    vmovq xmm10, qword ptr [r8 + r12*8]
    vmovhps xmm10, xmm10, qword ptr [r8 + r13*8]
    vinserti128 ymm8, ymm8, xmm10, 1
    vpbroadcastd ymm10, dword ptr [rax + 84]
    vpshufb ymm10, ymm10, ymm0
    vpand ymm10, ymm10, ymm1
    vpaddb ymm10, ymm10, ymm2
    vmovdqa ymm11, ymmword ptr [rdx + 320]
    vpsignb ymm11, ymm11, ymm10
    vpmaddubsw ymm11, ymm8, ymm11
    vmovdqa ymm12, ymmword ptr [rdx + 352]
    vpsignb ymm12, ymm12, ymm10
    vpmaddubsw ymm8, ymm8, ymm12
    # sums of group 5
    vpmaddwd ymm10, ymm11, ymm3
    vcvtdq2ps ymm10, ymm10
    vbroadcastss ymm11, dword ptr [rax + 40]
    vfmadd231ps ymm4, ymm10, ymm11
    vpmaddwd ymm8, ymm8, ymm3
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm10, dword ptr [rax + 44]
    vfmadd231ps ymm5, ymm8, ymm10
    vpand ymm9, ymm9, ymmword ptr [rip + {c8}]
    vpor ymm7, ymm7, ymm9
    vmovdqa ymmword ptr [r10 + 64], ymm7
    # products of group 6 of superblock +0
    movzx r11d, byte ptr [rdi + 50]
    movzx ebx, byte ptr [rdi + 51]
    movzx r12d, byte ptr [rdi + 52]
    movzx r13d, byte ptr [rdi + 53]
    vmovq xmm7, qword ptr [r8 + r11*8]
    vmovhps xmm7, xmm7, qword ptr [r8 + rbx*8]
    vmovq xmm8, qword ptr [r8 + r12*8]
    vmovhps xmm8, xmm8, qword ptr [r8 + r13*8]
    vinserti128 ymm7, ymm7, xmm8, 1
    vpbroadcastd ymm8, dword ptr [rax + 88]
    vpshufb ymm8, ymm8, ymm0
    vpand ymm8, ymm8, ymm1
    vpaddb ymm8, ymm8, ymm2
    vmovdqa ymm9, ymmword ptr [rdx + 384]
    vpsignb ymm9, ymm9, ymm8
    vpmaddubsw ymm9, ymm7, ymm9
    vmovdqa ymm10, ymmword ptr [rdx + 416]
    vpsignb ymm10, ymm10, ymm8
    vpmaddubsw ymm7, ymm7, ymm10
    # sums of group 6
    vpmaddwd ymm8, ymm9, ymm3
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm9, dword ptr [rax + 48]
    vfmadd231ps ymm4, ymm8, ymm9
    vpmaddwd ymm7, ymm7, ymm3
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rax + 52]
    vfmadd231ps ymm5, ymm7, ymm8
    # products of group 7 of superblock +0
    movzx r11d, byte ptr [rdi + 58]
    movzx ebx, byte ptr [rdi + 59]
    movzx r12d, byte ptr [rdi + 60]
    movzx r13d, byte ptr [rdi + 61]
    vmovq xmm7, qword ptr [r8 + r11*8]
    vmovhps xmm7, xmm7, qword ptr [r8 + rbx*8]
    vmovq xmm8, qword ptr [r8 + r12*8]
    vmovhps xmm8, xmm8, qword ptr [r8 + r13*8]
    vinserti128 ymm7, ymm7, xmm8, 1
    vpbroadcastd ymm8, dword ptr [rax + 92]
    vpshufb ymm8, ymm8, ymm0
    vpand ymm8, ymm8, ymm1
    vpaddb ymm8, ymm8, ymm2
    vmovdqa ymm9, ymmword ptr [rdx + 448]
    vpsignb ymm9, ymm9, ymm8
    vpmaddubsw ymm9, ymm7, ymm9
    vmovdqa ymm10, ymmword ptr [rdx + 480]
    vpsignb ymm10, ymm10, ymm8
    vpmaddubsw ymm7, ymm7, ymm10
    # sums of group 7
    vpmaddwd ymm8, ymm9, ymm3
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm9, dword ptr [rax + 56]
    vfmadd231ps ymm4, ymm8, ymm9
    vpmaddwd ymm7, ymm7, ymm3
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rax + 60]
    vfmadd231ps ymm5, ymm7, ymm8
    # pair scales of superblock +1
    vpbroadcastw xmm7, word ptr [rdi + 66]
    vcvtph2ps ymm7, xmm7
    vpor ymm6, ymm6, ymmword ptr [rip + {c9}]
    vcvtdq2ps ymm6, ymm6
    vmulps ymm6, ymm7, ymm6
    vmulps ymm6, ymm6, ymmword ptr [rip + {c10}]
    vpermq ymm6, ymm6, 0xd8
    vshufps ymm7, ymm6, ymm6, 0x50
    vmulps ymm7, ymm7, ymmword ptr [rcx + 64]
    vmovaps ymmword ptr [r10], ymm7
    vshufps ymm6, ymm6, ymm6, 0xfa
    vmulps ymm6, ymm6, ymmword ptr [rcx + 96]
    vmovaps ymmword ptr [r10 + 32], ymm6
    # products of group 0 of superblock +1
    movzx r11d, byte ptr [rdi + 68]
    movzx ebx, byte ptr [rdi + 69]
    movzx r12d, byte ptr [rdi + 70]
    movzx r13d, byte ptr [rdi + 71]
    vmovq xmm6, qword ptr [r8 + r11*8]
    vmovhps xmm6, xmm6, qword ptr [r8 + rbx*8]
    vmovq xmm7, qword ptr [r8 + r12*8]
    vmovhps xmm7, xmm7, qword ptr [r8 + r13*8]
    vinserti128 ymm6, ymm6, xmm7, 1
    vpbroadcastd ymm7, dword ptr [r10 + 64]
    vpshufb ymm7, ymm7, ymm0
    vpand ymm7, ymm7, ymm1
    vpaddb ymm7, ymm7, ymm2
    vmovdqa ymm8, ymmword ptr [rdx + 512]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm8, ymm6, ymm8
    vmovdqa ymm9, ymmword ptr [rdx + 544]
    vpsignb ymm9, ymm9, ymm7
    vpmaddubsw ymm6, ymm6, ymm9
    add rdi, 66
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
    vfmadd231ps ymm4, ymm7, ymm8
    vpmaddwd ymm6, ymm6, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 4]
    vfmadd231ps ymm5, ymm6, ymm7
    # products of group 1 of superblock +0
    movzx ecx, byte ptr [rdi + 10]
    movzx esi, byte ptr [rdi + 11]
    movzx r10d, byte ptr [rdi + 12]
    movzx r11d, byte ptr [rdi + 13]
    vmovq xmm6, qword ptr [r8 + rcx*8]
    vmovhps xmm6, xmm6, qword ptr [r8 + rsi*8]
    vmovq xmm7, qword ptr [r8 + r10*8]
    vmovhps xmm7, xmm7, qword ptr [r8 + r11*8]
    vinserti128 ymm6, ymm6, xmm7, 1
    vpbroadcastd ymm7, dword ptr [rax + 68]
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
    vfmadd231ps ymm4, ymm6, ymm7
    vpmaddwd ymm6, ymm8, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 12]
    vfmadd231ps ymm5, ymm6, ymm7
    # products of group 2 of superblock +0
    movzx ecx, byte ptr [rdi + 18]
    movzx esi, byte ptr [rdi + 19]
    movzx r10d, byte ptr [rdi + 20]
    movzx r11d, byte ptr [rdi + 21]
    vmovq xmm6, qword ptr [r8 + rcx*8]
    vmovhps xmm6, xmm6, qword ptr [r8 + rsi*8]
    vmovq xmm7, qword ptr [r8 + r10*8]
    vmovhps xmm7, xmm7, qword ptr [r8 + r11*8]
    vinserti128 ymm6, ymm6, xmm7, 1
    vpbroadcastd ymm7, dword ptr [rax + 72]
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
    vfmadd231ps ymm4, ymm6, ymm7
    vpmaddwd ymm6, ymm9, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 20]
    vfmadd231ps ymm5, ymm6, ymm7
    # products of group 3 of superblock +0
    movzx ecx, byte ptr [rdi + 26]
    movzx esi, byte ptr [rdi + 27]
    movzx r10d, byte ptr [rdi + 28]
    movzx r11d, byte ptr [rdi + 29]
    vmovq xmm6, qword ptr [r8 + rcx*8]
    vmovhps xmm6, xmm6, qword ptr [r8 + rsi*8]
    vmovq xmm7, qword ptr [r8 + r10*8]
    vmovhps xmm7, xmm7, qword ptr [r8 + r11*8]
    vinserti128 ymm6, ymm6, xmm7, 1
    vpbroadcastd ymm7, dword ptr [rax + 76]
    vpshufb ymm7, ymm7, ymm0
    vpand ymm7, ymm7, ymm1
    vpaddb ymm7, ymm7, ymm2
    vmovdqa ymm8, ymmword ptr [rdx + 192]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm11, ymm6, ymm8
    vmovdqa ymm8, ymmword ptr [rdx + 224]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm9, ymm6, ymm8
    # sums of group 3
    vpmaddwd ymm6, ymm11, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 24]
    vfmadd231ps ymm4, ymm6, ymm7
    vpmaddwd ymm6, ymm9, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 28]
    vfmadd231ps ymm5, ymm6, ymm7
    # products of group 4 of superblock +0
    movzx ecx, byte ptr [rdi + 34]
    movzx esi, byte ptr [rdi + 35]
    movzx r10d, byte ptr [rdi + 36]
    movzx r11d, byte ptr [rdi + 37]
    vmovq xmm6, qword ptr [r8 + rcx*8]
    vmovhps xmm6, xmm6, qword ptr [r8 + rsi*8]
    vmovq xmm7, qword ptr [r8 + r10*8]
    vmovhps xmm7, xmm7, qword ptr [r8 + r11*8]
    vinserti128 ymm6, ymm6, xmm7, 1
    vpbroadcastd ymm7, dword ptr [rax + 80]
    vpshufb ymm7, ymm7, ymm0
    vpand ymm7, ymm7, ymm1
    vpaddb ymm7, ymm7, ymm2
    vmovdqa ymm8, ymmword ptr [rdx + 256]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm11, ymm6, ymm8
    vmovdqa ymm8, ymmword ptr [rdx + 288]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm8, ymm6, ymm8
    # sums of group 4
    vpmaddwd ymm6, ymm11, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 32]
    vfmadd231ps ymm4, ymm6, ymm7
    vpmaddwd ymm6, ymm8, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 36]
    vfmadd231ps ymm5, ymm6, ymm7
    # products of group 5 of superblock +0
    movzx ecx, byte ptr [rdi + 42]
    movzx esi, byte ptr [rdi + 43]
    movzx r10d, byte ptr [rdi + 44]
    movzx r11d, byte ptr [rdi + 45]
    vmovq xmm6, qword ptr [r8 + rcx*8]
    vmovhps xmm6, xmm6, qword ptr [r8 + rsi*8]
    vmovq xmm7, qword ptr [r8 + r10*8]
    vmovhps xmm7, xmm7, qword ptr [r8 + r11*8]
    vinserti128 ymm6, ymm6, xmm7, 1
    vpbroadcastd ymm7, dword ptr [rax + 84]
    vpshufb ymm7, ymm7, ymm0
    vpand ymm7, ymm7, ymm1
    vpaddb ymm7, ymm7, ymm2
    vmovdqa ymm8, ymmword ptr [rdx + 320]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm11, ymm6, ymm8
    vmovdqa ymm8, ymmword ptr [rdx + 352]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm8, ymm6, ymm8
    # sums of group 5
    vpmaddwd ymm6, ymm11, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 40]
    vfmadd231ps ymm4, ymm6, ymm7
    vpmaddwd ymm6, ymm8, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 44]
    vfmadd231ps ymm5, ymm6, ymm7
    # products of group 6 of superblock +0
    movzx ecx, byte ptr [rdi + 50]
    movzx esi, byte ptr [rdi + 51]
    movzx r10d, byte ptr [rdi + 52]
    movzx r11d, byte ptr [rdi + 53]
    vmovq xmm6, qword ptr [r8 + rcx*8]
    vmovhps xmm6, xmm6, qword ptr [r8 + rsi*8]
    vmovq xmm7, qword ptr [r8 + r10*8]
    vmovhps xmm7, xmm7, qword ptr [r8 + r11*8]
    vinserti128 ymm6, ymm6, xmm7, 1
    vpbroadcastd ymm7, dword ptr [rax + 88]
    vpshufb ymm7, ymm7, ymm0
    vpand ymm7, ymm7, ymm1
    vpaddb ymm7, ymm7, ymm2
    vmovdqa ymm8, ymmword ptr [rdx + 384]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm9, ymm6, ymm8
    vmovdqa ymm8, ymmword ptr [rdx + 416]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm7, ymm6, ymm8
    # sums of group 6
    vpmaddwd ymm6, ymm9, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm8, dword ptr [rax + 48]
    vfmadd231ps ymm4, ymm6, ymm8
    vpmaddwd ymm6, ymm7, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 52]
    vfmadd231ps ymm5, ymm6, ymm7
    # products of group 7 of superblock +0
    movzx ecx, byte ptr [rdi + 58]
    movzx esi, byte ptr [rdi + 59]
    movzx r10d, byte ptr [rdi + 60]
    movzx edi, byte ptr [rdi + 61]
    vmovq xmm6, qword ptr [r8 + rcx*8]
    vmovhps xmm6, xmm6, qword ptr [r8 + rsi*8]
    vmovq xmm7, qword ptr [r8 + r10*8]
    vmovhps xmm7, xmm7, qword ptr [r8 + rdi*8]
    vinserti128 ymm6, ymm6, xmm7, 1
    vpbroadcastd ymm7, dword ptr [rax + 92]
    vpshufb ymm7, ymm7, ymm0
    vpand ymm7, ymm7, ymm1
    vpaddb ymm7, ymm7, ymm2
    vmovdqa ymm0, ymmword ptr [rdx + 448]
    vpsignb ymm0, ymm0, ymm7
    vpmaddubsw ymm9, ymm6, ymm0
    vmovdqa ymm0, ymmword ptr [rdx + 480]
    vpsignb ymm0, ymm0, ymm7
    vpmaddubsw ymm7, ymm6, ymm0
    # sums of group 7
    vpmaddwd ymm0, ymm9, ymm3
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm1, dword ptr [rax + 56]
    vfmadd231ps ymm4, ymm0, ymm1
    vpmaddwd ymm0, ymm7, ymm3
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm1, dword ptr [rax + 60]
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
#   7f 00 00 00 7f 00 00 00 7f 00 00 00 7f 00 00 00
#   7f 00 00 00 7f 00 00 00 7f 00 00 00 7f 00 00 00
# c5, aligned to 32:
#   00 7f 00 00 00 7f 00 00 00 7f 00 00 00 7f 00 00
#   00 7f 00 00 00 7f 00 00 00 7f 00 00 00 7f 00 00
# c6, aligned to 32:
#   00 00 7f 00 00 00 7f 00 00 00 7f 00 00 00 7f 00
#   00 00 7f 00 00 00 7f 00 00 00 7f 00 00 00 7f 00
# c7, aligned to 32:
#   00 00 00 7f 00 00 00 7f 00 00 00 7f 00 00 00 7f
#   00 00 00 7f 00 00 00 7f 00 00 00 7f 00 00 00 7f
# c8, aligned to 32:
#   80 80 80 80 80 80 80 80 80 80 80 80 80 80 80 80
#   80 80 80 80 80 80 80 80 80 80 80 80 80 80 80 80
# c9, aligned to 32:
#   01 00 00 00 01 00 00 00 01 00 00 00 01 00 00 00
#   01 00 00 00 01 00 00 00 01 00 00 00 01 00 00 00
# c10, aligned to 32:
#   00 00 00 3e 00 00 00 3e 00 00 00 3e 00 00 00 3e
#   00 00 00 3e 00 00 00 3e 00 00 00 3e 00 00 00 3e
