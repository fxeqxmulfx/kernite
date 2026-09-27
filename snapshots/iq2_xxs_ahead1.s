# iq2_xxs_ahead1(gpr, gpr, gpr, gpr, gpr) -> vec
# frame: 128 bytes aligned to 128, saves [rbx, r12]
# row=rdi sbs=rsi quants=rdx scales=rcx grid=r8 spread=ymm0
# bit=ymm1 one8=ymm2 one16=ymm3 acc=ymm4 cur=rax nxt=r9
# p0=ymm5 p1=ymm8 p2=ymm7 p3=ymm9 p4=ymm6 p5=ymm9
# p6=ymm6 p7=ymm7 words=ymm5,ymm6 d=ymm6,ymm7 scale=ymm7,ymm5 fields=ymm6,ymm5
# field=ymm7,ymm8 parity=ymm5,ymm7 index0=r10,rcx index1=r11,rsi index2=rbx,r9 index3=r12,r10,rdi
# rows=ymm6,ymm9,ymm10,ymm7,ymm5 signs=ymm6,ymm9,ymm10,ymm7,ymm5 x=ymm7,ymm10,ymm11,ymm8,ymm6,ymm0 lanes=ymm5,ymm8,ymm7,ymm6,ymm0 group_scale=ymm9,ymm10,ymm8,ymm6,ymm7,ymm1 zero=ymm0
# high=ymm0 sum=ymm0 pair=ymm1 result=ymm0 nothing=ymm0
    push rbx
    push r12
    push rbp
    mov rbp, rsp
    and rsp, -128
    sub rsp, 128
    test rsi, rsi
    jz 4f
    # constants
    vmovdqa ymm0, ymmword ptr [rip + {c0}]
    vpbroadcastq ymm1, qword ptr [rip + {c1}]
    vpbroadcastd ymm2, dword ptr [rip + {c2}]
    vpbroadcastd ymm3, dword ptr [rip + {c3}]
    vxorps xmm4, xmm4, xmm4
    lea rax, [rsp]
    lea r9, [rsp + 64]
    # scales and signs of superblock +0
    vmovdqu ymm5, ymmword ptr [rdi + 2]
    vshufps ymm5, ymm5, ymmword ptr [rdi + 34], 0xdd
    vpermq ymm5, ymm5, 0xd8
    vpbroadcastw xmm6, word ptr [rdi]
    vcvtph2ps ymm6, xmm6
    vpsrld ymm7, ymm5, 27
    vpor ymm7, ymm7, ymmword ptr [rip + {c9}]
    vcvtdq2ps ymm7, ymm7
    vmulps ymm7, ymm6, ymm7
    vmulps ymm7, ymm7, ymmword ptr [rip + {c10}]
    vmulps ymm7, ymm7, ymmword ptr [rcx]
    vmovaps ymmword ptr [rax], ymm7
    vpand ymm6, ymm5, ymmword ptr [rip + {c4}]
    vpaddd ymm7, ymm5, ymm5
    vpand ymm7, ymm7, ymmword ptr [rip + {c5}]
    vpor ymm6, ymm6, ymm7
    vpslld ymm7, ymm5, 2
    vpand ymm7, ymm7, ymmword ptr [rip + {c6}]
    vpor ymm6, ymm6, ymm7
    vpslld ymm7, ymm5, 3
    vpand ymm7, ymm7, ymmword ptr [rip + {c7}]
    vpor ymm6, ymm6, ymm7
    vpaddw ymm5, ymm6, ymm6
    vpxor ymm5, ymm5, ymm6
    vpsllw ymm7, ymm5, 2
    vpxor ymm5, ymm5, ymm7
    vpsllw ymm7, ymm5, 4
    vpxor ymm5, ymm5, ymm7
    vpand ymm5, ymm5, ymmword ptr [rip + {c8}]
    vpor ymm6, ymm6, ymm5
    vmovdqa ymmword ptr [rax + 32], ymm6
    # products of group 0 of superblock +0
    movzx r10d, byte ptr [rdi + 2]
    movzx r11d, byte ptr [rdi + 3]
    movzx ebx, byte ptr [rdi + 4]
    movzx r12d, byte ptr [rdi + 5]
    vmovq xmm5, qword ptr [r8 + r10*8]
    vpbroadcastq ymm6, qword ptr [r8 + r11*8]
    vpblendd ymm5, ymm5, ymm6, 0x0c
    vpbroadcastq ymm6, qword ptr [r8 + rbx*8]
    vpblendd ymm5, ymm5, ymm6, 0x30
    vpbroadcastq ymm6, qword ptr [r8 + r12*8]
    vpblendd ymm5, ymm5, ymm6, 0xc0
    vpbroadcastd ymm6, dword ptr [rax + 32]
    vpshufb ymm6, ymm6, ymm0
    vpand ymm6, ymm6, ymm1
    vpcmpeqb ymm6, ymm6, ymm1
    vpor ymm6, ymm6, ymm2
    vmovdqa ymm7, ymmword ptr [rdx]
    vpsignb ymm7, ymm7, ymm6
    vpmaddubsw ymm5, ymm5, ymm7
    dec rsi
    jz 3f
2:
    # scales and signs of superblock +1
    vmovdqu ymm6, ymmword ptr [rdi + 68]
    vshufps ymm6, ymm6, ymmword ptr [rdi + 100], 0xdd
    vpermq ymm6, ymm6, 0xd8
    vpbroadcastw xmm7, word ptr [rdi + 66]
    vcvtph2ps ymm7, xmm7
    # products of group 1 of superblock +0
    movzx r10d, byte ptr [rdi + 10]
    movzx r11d, byte ptr [rdi + 11]
    movzx ebx, byte ptr [rdi + 12]
    movzx r12d, byte ptr [rdi + 13]
    vmovq xmm8, qword ptr [r8 + r10*8]
    vpbroadcastq ymm9, qword ptr [r8 + r11*8]
    vpblendd ymm8, ymm8, ymm9, 0x0c
    vpbroadcastq ymm9, qword ptr [r8 + rbx*8]
    vpblendd ymm8, ymm8, ymm9, 0x30
    vpbroadcastq ymm9, qword ptr [r8 + r12*8]
    vpblendd ymm8, ymm8, ymm9, 0xc0
    vpbroadcastd ymm9, dword ptr [rax + 36]
    vpshufb ymm9, ymm9, ymm0
    vpand ymm9, ymm9, ymm1
    vpcmpeqb ymm9, ymm9, ymm1
    vpor ymm9, ymm9, ymm2
    vmovdqa ymm10, ymmword ptr [rdx + 32]
    vpsignb ymm10, ymm10, ymm9
    vpmaddubsw ymm8, ymm8, ymm10
    # sum of group 0
    vpmaddwd ymm5, ymm5, ymm3
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm9, dword ptr [rax]
    vfmadd231ps ymm4, ymm5, ymm9
    vpsrld ymm5, ymm6, 27
    vpor ymm5, ymm5, ymmword ptr [rip + {c9}]
    vcvtdq2ps ymm5, ymm5
    vmulps ymm5, ymm7, ymm5
    vmulps ymm5, ymm5, ymmword ptr [rip + {c10}]
    # products of group 2 of superblock +0
    movzx r10d, byte ptr [rdi + 18]
    movzx r11d, byte ptr [rdi + 19]
    movzx ebx, byte ptr [rdi + 20]
    movzx r12d, byte ptr [rdi + 21]
    vmovq xmm7, qword ptr [r8 + r10*8]
    vpbroadcastq ymm9, qword ptr [r8 + r11*8]
    vpblendd ymm7, ymm7, ymm9, 0x0c
    vpbroadcastq ymm9, qword ptr [r8 + rbx*8]
    vpblendd ymm7, ymm7, ymm9, 0x30
    vpbroadcastq ymm9, qword ptr [r8 + r12*8]
    vpblendd ymm7, ymm7, ymm9, 0xc0
    vpbroadcastd ymm9, dword ptr [rax + 40]
    vpshufb ymm9, ymm9, ymm0
    vpand ymm9, ymm9, ymm1
    vpcmpeqb ymm9, ymm9, ymm1
    vpor ymm9, ymm9, ymm2
    vmovdqa ymm10, ymmword ptr [rdx + 64]
    vpsignb ymm10, ymm10, ymm9
    vpmaddubsw ymm7, ymm7, ymm10
    # sum of group 1
    vpmaddwd ymm8, ymm8, ymm3
    vcvtdq2ps ymm8, ymm8
    vbroadcastss ymm9, dword ptr [rax + 4]
    vfmadd231ps ymm4, ymm8, ymm9
    vmulps ymm5, ymm5, ymmword ptr [rcx + 32]
    vmovaps ymmword ptr [r9], ymm5
    vpand ymm5, ymm6, ymmword ptr [rip + {c4}]
    vpaddd ymm8, ymm6, ymm6
    vpand ymm8, ymm8, ymmword ptr [rip + {c5}]
    # products of group 3 of superblock +0
    movzx r10d, byte ptr [rdi + 26]
    movzx r11d, byte ptr [rdi + 27]
    movzx ebx, byte ptr [rdi + 28]
    movzx r12d, byte ptr [rdi + 29]
    vmovq xmm9, qword ptr [r8 + r10*8]
    vpbroadcastq ymm10, qword ptr [r8 + r11*8]
    vpblendd ymm9, ymm9, ymm10, 0x0c
    vpbroadcastq ymm10, qword ptr [r8 + rbx*8]
    vpblendd ymm9, ymm9, ymm10, 0x30
    vpbroadcastq ymm10, qword ptr [r8 + r12*8]
    vpblendd ymm9, ymm9, ymm10, 0xc0
    vpbroadcastd ymm10, dword ptr [rax + 44]
    vpshufb ymm10, ymm10, ymm0
    vpand ymm10, ymm10, ymm1
    vpcmpeqb ymm10, ymm10, ymm1
    vpor ymm10, ymm10, ymm2
    vmovdqa ymm11, ymmword ptr [rdx + 96]
    vpsignb ymm11, ymm11, ymm10
    vpmaddubsw ymm9, ymm9, ymm11
    # sum of group 2
    vpmaddwd ymm7, ymm7, ymm3
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm10, dword ptr [rax + 8]
    vfmadd231ps ymm4, ymm7, ymm10
    vpor ymm5, ymm5, ymm8
    vpslld ymm8, ymm6, 2
    vpand ymm8, ymm8, ymmword ptr [rip + {c6}]
    vpor ymm5, ymm5, ymm8
    vpslld ymm8, ymm6, 3
    # products of group 4 of superblock +0
    movzx r10d, byte ptr [rdi + 34]
    movzx r11d, byte ptr [rdi + 35]
    movzx ebx, byte ptr [rdi + 36]
    movzx r12d, byte ptr [rdi + 37]
    vmovq xmm6, qword ptr [r8 + r10*8]
    vpbroadcastq ymm7, qword ptr [r8 + r11*8]
    vpblendd ymm6, ymm6, ymm7, 0x0c
    vpbroadcastq ymm7, qword ptr [r8 + rbx*8]
    vpblendd ymm6, ymm6, ymm7, 0x30
    vpbroadcastq ymm7, qword ptr [r8 + r12*8]
    vpblendd ymm6, ymm6, ymm7, 0xc0
    vpbroadcastd ymm7, dword ptr [rax + 48]
    vpshufb ymm7, ymm7, ymm0
    vpand ymm7, ymm7, ymm1
    vpcmpeqb ymm7, ymm7, ymm1
    vpor ymm7, ymm7, ymm2
    vmovdqa ymm10, ymmword ptr [rdx + 128]
    vpsignb ymm10, ymm10, ymm7
    vpmaddubsw ymm6, ymm6, ymm10
    # sum of group 3
    vpmaddwd ymm7, ymm9, ymm3
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm9, dword ptr [rax + 12]
    vfmadd231ps ymm4, ymm7, ymm9
    vpand ymm8, ymm8, ymmword ptr [rip + {c7}]
    vpor ymm5, ymm5, ymm8
    vpaddw ymm7, ymm5, ymm5
    vpxor ymm7, ymm7, ymm5
    vpsllw ymm8, ymm7, 2
    # products of group 5 of superblock +0
    movzx r10d, byte ptr [rdi + 42]
    movzx r11d, byte ptr [rdi + 43]
    movzx ebx, byte ptr [rdi + 44]
    movzx r12d, byte ptr [rdi + 45]
    vmovq xmm9, qword ptr [r8 + r10*8]
    vpbroadcastq ymm10, qword ptr [r8 + r11*8]
    vpblendd ymm9, ymm9, ymm10, 0x0c
    vpbroadcastq ymm10, qword ptr [r8 + rbx*8]
    vpblendd ymm9, ymm9, ymm10, 0x30
    vpbroadcastq ymm10, qword ptr [r8 + r12*8]
    vpblendd ymm9, ymm9, ymm10, 0xc0
    vpbroadcastd ymm10, dword ptr [rax + 52]
    vpshufb ymm10, ymm10, ymm0
    vpand ymm10, ymm10, ymm1
    vpcmpeqb ymm10, ymm10, ymm1
    vpor ymm10, ymm10, ymm2
    vmovdqa ymm11, ymmword ptr [rdx + 160]
    vpsignb ymm11, ymm11, ymm10
    vpmaddubsw ymm9, ymm9, ymm11
    # sum of group 4
    vpmaddwd ymm6, ymm6, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm10, dword ptr [rax + 16]
    vfmadd231ps ymm4, ymm6, ymm10
    vpxor ymm7, ymm7, ymm8
    vpsllw ymm8, ymm7, 4
    vpxor ymm7, ymm7, ymm8
    vpand ymm7, ymm7, ymmword ptr [rip + {c8}]
    vpor ymm5, ymm5, ymm7
    # products of group 6 of superblock +0
    movzx r10d, byte ptr [rdi + 50]
    movzx r11d, byte ptr [rdi + 51]
    movzx ebx, byte ptr [rdi + 52]
    movzx r12d, byte ptr [rdi + 53]
    vmovq xmm6, qword ptr [r8 + r10*8]
    vpbroadcastq ymm7, qword ptr [r8 + r11*8]
    vpblendd ymm6, ymm6, ymm7, 0x0c
    vpbroadcastq ymm7, qword ptr [r8 + rbx*8]
    vpblendd ymm6, ymm6, ymm7, 0x30
    vpbroadcastq ymm7, qword ptr [r8 + r12*8]
    vpblendd ymm6, ymm6, ymm7, 0xc0
    vpbroadcastd ymm7, dword ptr [rax + 56]
    vpshufb ymm7, ymm7, ymm0
    vpand ymm7, ymm7, ymm1
    vpcmpeqb ymm7, ymm7, ymm1
    vpor ymm7, ymm7, ymm2
    vmovdqa ymm8, ymmword ptr [rdx + 192]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm6, ymm6, ymm8
    # sum of group 5
    vpmaddwd ymm7, ymm9, ymm3
    vcvtdq2ps ymm7, ymm7
    vbroadcastss ymm8, dword ptr [rax + 20]
    vfmadd231ps ymm4, ymm7, ymm8
    vmovdqa ymmword ptr [r9 + 32], ymm5
    # products of group 7 of superblock +0
    movzx r10d, byte ptr [rdi + 58]
    movzx r11d, byte ptr [rdi + 59]
    movzx ebx, byte ptr [rdi + 60]
    movzx r12d, byte ptr [rdi + 61]
    vmovq xmm7, qword ptr [r8 + r10*8]
    vpbroadcastq ymm5, qword ptr [r8 + r11*8]
    vpblendd ymm7, ymm7, ymm5, 0x0c
    vpbroadcastq ymm5, qword ptr [r8 + rbx*8]
    vpblendd ymm7, ymm7, ymm5, 0x30
    vpbroadcastq ymm5, qword ptr [r8 + r12*8]
    vpblendd ymm7, ymm7, ymm5, 0xc0
    vpbroadcastd ymm5, dword ptr [rax + 60]
    vpshufb ymm5, ymm5, ymm0
    vpand ymm5, ymm5, ymm1
    vpcmpeqb ymm5, ymm5, ymm1
    vpor ymm5, ymm5, ymm2
    vmovdqa ymm8, ymmword ptr [rdx + 224]
    vpsignb ymm8, ymm8, ymm5
    vpmaddubsw ymm7, ymm7, ymm8
    # sum of group 6
    vpmaddwd ymm5, ymm6, ymm3
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 24]
    vfmadd231ps ymm4, ymm5, ymm6
    # products of group 0 of superblock +1
    movzx r10d, byte ptr [rdi + 68]
    movzx r11d, byte ptr [rdi + 69]
    movzx ebx, byte ptr [rdi + 70]
    movzx r12d, byte ptr [rdi + 71]
    vmovq xmm5, qword ptr [r8 + r10*8]
    vpbroadcastq ymm6, qword ptr [r8 + r11*8]
    vpblendd ymm5, ymm5, ymm6, 0x0c
    vpbroadcastq ymm6, qword ptr [r8 + rbx*8]
    vpblendd ymm5, ymm5, ymm6, 0x30
    vpbroadcastq ymm6, qword ptr [r8 + r12*8]
    vpblendd ymm5, ymm5, ymm6, 0xc0
    vpbroadcastd ymm6, dword ptr [r9 + 32]
    vpshufb ymm6, ymm6, ymm0
    vpand ymm6, ymm6, ymm1
    vpcmpeqb ymm6, ymm6, ymm1
    vpor ymm6, ymm6, ymm2
    vmovdqa ymm8, ymmword ptr [rdx + 256]
    vpsignb ymm8, ymm8, ymm6
    vpmaddubsw ymm5, ymm5, ymm8
    # sum of group 7
    vpmaddwd ymm6, ymm7, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 28]
    vfmadd231ps ymm4, ymm6, ymm7
    add rdi, 66
    add rdx, 256
    add rcx, 32
    xor rax, 64
    xor r9, 64
    dec rsi
    jnz 2b
3:
    # products of group 1 of superblock +0
    movzx ecx, byte ptr [rdi + 10]
    movzx esi, byte ptr [rdi + 11]
    movzx r9d, byte ptr [rdi + 12]
    movzx r10d, byte ptr [rdi + 13]
    vmovq xmm8, qword ptr [r8 + rcx*8]
    vpbroadcastq ymm6, qword ptr [r8 + rsi*8]
    vpblendd ymm8, ymm8, ymm6, 0x0c
    vpbroadcastq ymm6, qword ptr [r8 + r9*8]
    vpblendd ymm8, ymm8, ymm6, 0x30
    vpbroadcastq ymm6, qword ptr [r8 + r10*8]
    vpblendd ymm8, ymm8, ymm6, 0xc0
    vpbroadcastd ymm6, dword ptr [rax + 36]
    vpshufb ymm6, ymm6, ymm0
    vpand ymm6, ymm6, ymm1
    vpcmpeqb ymm6, ymm6, ymm1
    vpor ymm6, ymm6, ymm2
    vmovdqa ymm7, ymmword ptr [rdx + 32]
    vpsignb ymm7, ymm7, ymm6
    vpmaddubsw ymm8, ymm8, ymm7
    # sum of group 0
    vpmaddwd ymm5, ymm5, ymm3
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax]
    vfmadd231ps ymm4, ymm5, ymm6
    # products of group 2 of superblock +0
    movzx ecx, byte ptr [rdi + 18]
    movzx esi, byte ptr [rdi + 19]
    movzx r9d, byte ptr [rdi + 20]
    movzx r10d, byte ptr [rdi + 21]
    vmovq xmm7, qword ptr [r8 + rcx*8]
    vpbroadcastq ymm5, qword ptr [r8 + rsi*8]
    vpblendd ymm7, ymm7, ymm5, 0x0c
    vpbroadcastq ymm5, qword ptr [r8 + r9*8]
    vpblendd ymm7, ymm7, ymm5, 0x30
    vpbroadcastq ymm5, qword ptr [r8 + r10*8]
    vpblendd ymm7, ymm7, ymm5, 0xc0
    vpbroadcastd ymm5, dword ptr [rax + 40]
    vpshufb ymm5, ymm5, ymm0
    vpand ymm5, ymm5, ymm1
    vpcmpeqb ymm5, ymm5, ymm1
    vpor ymm5, ymm5, ymm2
    vmovdqa ymm6, ymmword ptr [rdx + 64]
    vpsignb ymm6, ymm6, ymm5
    vpmaddubsw ymm7, ymm7, ymm6
    # sum of group 1
    vpmaddwd ymm5, ymm8, ymm3
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 4]
    vfmadd231ps ymm4, ymm5, ymm6
    # products of group 3 of superblock +0
    movzx ecx, byte ptr [rdi + 26]
    movzx esi, byte ptr [rdi + 27]
    movzx r9d, byte ptr [rdi + 28]
    movzx r10d, byte ptr [rdi + 29]
    vmovq xmm9, qword ptr [r8 + rcx*8]
    vpbroadcastq ymm5, qword ptr [r8 + rsi*8]
    vpblendd ymm9, ymm9, ymm5, 0x0c
    vpbroadcastq ymm5, qword ptr [r8 + r9*8]
    vpblendd ymm9, ymm9, ymm5, 0x30
    vpbroadcastq ymm5, qword ptr [r8 + r10*8]
    vpblendd ymm9, ymm9, ymm5, 0xc0
    vpbroadcastd ymm5, dword ptr [rax + 44]
    vpshufb ymm5, ymm5, ymm0
    vpand ymm5, ymm5, ymm1
    vpcmpeqb ymm5, ymm5, ymm1
    vpor ymm5, ymm5, ymm2
    vmovdqa ymm6, ymmword ptr [rdx + 96]
    vpsignb ymm6, ymm6, ymm5
    vpmaddubsw ymm9, ymm9, ymm6
    # sum of group 2
    vpmaddwd ymm5, ymm7, ymm3
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 8]
    vfmadd231ps ymm4, ymm5, ymm6
    # products of group 4 of superblock +0
    movzx ecx, byte ptr [rdi + 34]
    movzx esi, byte ptr [rdi + 35]
    movzx r9d, byte ptr [rdi + 36]
    movzx r10d, byte ptr [rdi + 37]
    vmovq xmm6, qword ptr [r8 + rcx*8]
    vpbroadcastq ymm5, qword ptr [r8 + rsi*8]
    vpblendd ymm6, ymm6, ymm5, 0x0c
    vpbroadcastq ymm5, qword ptr [r8 + r9*8]
    vpblendd ymm6, ymm6, ymm5, 0x30
    vpbroadcastq ymm5, qword ptr [r8 + r10*8]
    vpblendd ymm6, ymm6, ymm5, 0xc0
    vpbroadcastd ymm5, dword ptr [rax + 48]
    vpshufb ymm5, ymm5, ymm0
    vpand ymm5, ymm5, ymm1
    vpcmpeqb ymm5, ymm5, ymm1
    vpor ymm5, ymm5, ymm2
    vmovdqa ymm7, ymmword ptr [rdx + 128]
    vpsignb ymm7, ymm7, ymm5
    vpmaddubsw ymm6, ymm6, ymm7
    # sum of group 3
    vpmaddwd ymm5, ymm9, ymm3
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm7, dword ptr [rax + 12]
    vfmadd231ps ymm4, ymm5, ymm7
    # products of group 5 of superblock +0
    movzx ecx, byte ptr [rdi + 42]
    movzx esi, byte ptr [rdi + 43]
    movzx r9d, byte ptr [rdi + 44]
    movzx r10d, byte ptr [rdi + 45]
    vmovq xmm9, qword ptr [r8 + rcx*8]
    vpbroadcastq ymm5, qword ptr [r8 + rsi*8]
    vpblendd ymm9, ymm9, ymm5, 0x0c
    vpbroadcastq ymm5, qword ptr [r8 + r9*8]
    vpblendd ymm9, ymm9, ymm5, 0x30
    vpbroadcastq ymm5, qword ptr [r8 + r10*8]
    vpblendd ymm9, ymm9, ymm5, 0xc0
    vpbroadcastd ymm5, dword ptr [rax + 52]
    vpshufb ymm5, ymm5, ymm0
    vpand ymm5, ymm5, ymm1
    vpcmpeqb ymm5, ymm5, ymm1
    vpor ymm5, ymm5, ymm2
    vmovdqa ymm7, ymmword ptr [rdx + 160]
    vpsignb ymm7, ymm7, ymm5
    vpmaddubsw ymm9, ymm9, ymm7
    # sum of group 4
    vpmaddwd ymm5, ymm6, ymm3
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax + 16]
    vfmadd231ps ymm4, ymm5, ymm6
    # products of group 6 of superblock +0
    movzx ecx, byte ptr [rdi + 50]
    movzx esi, byte ptr [rdi + 51]
    movzx r9d, byte ptr [rdi + 52]
    movzx r10d, byte ptr [rdi + 53]
    vmovq xmm6, qword ptr [r8 + rcx*8]
    vpbroadcastq ymm5, qword ptr [r8 + rsi*8]
    vpblendd ymm6, ymm6, ymm5, 0x0c
    vpbroadcastq ymm5, qword ptr [r8 + r9*8]
    vpblendd ymm6, ymm6, ymm5, 0x30
    vpbroadcastq ymm5, qword ptr [r8 + r10*8]
    vpblendd ymm6, ymm6, ymm5, 0xc0
    vpbroadcastd ymm5, dword ptr [rax + 56]
    vpshufb ymm5, ymm5, ymm0
    vpand ymm5, ymm5, ymm1
    vpcmpeqb ymm5, ymm5, ymm1
    vpor ymm5, ymm5, ymm2
    vmovdqa ymm7, ymmword ptr [rdx + 192]
    vpsignb ymm7, ymm7, ymm5
    vpmaddubsw ymm6, ymm6, ymm7
    # sum of group 5
    vpmaddwd ymm5, ymm9, ymm3
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm7, dword ptr [rax + 20]
    vfmadd231ps ymm4, ymm5, ymm7
    # products of group 7 of superblock +0
    movzx ecx, byte ptr [rdi + 58]
    movzx esi, byte ptr [rdi + 59]
    movzx r9d, byte ptr [rdi + 60]
    movzx edi, byte ptr [rdi + 61]
    vmovq xmm7, qword ptr [r8 + rcx*8]
    vpbroadcastq ymm5, qword ptr [r8 + rsi*8]
    vpblendd ymm7, ymm7, ymm5, 0x0c
    vpbroadcastq ymm5, qword ptr [r8 + r9*8]
    vpblendd ymm7, ymm7, ymm5, 0x30
    vpbroadcastq ymm5, qword ptr [r8 + rdi*8]
    vpblendd ymm7, ymm7, ymm5, 0xc0
    vpbroadcastd ymm5, dword ptr [rax + 60]
    vpshufb ymm5, ymm5, ymm0
    vpand ymm5, ymm5, ymm1
    vpcmpeqb ymm5, ymm5, ymm1
    vpor ymm5, ymm5, ymm2
    vmovdqa ymm0, ymmword ptr [rdx + 224]
    vpsignb ymm0, ymm0, ymm5
    vpmaddubsw ymm7, ymm7, ymm0
    # sum of group 6
    vpmaddwd ymm0, ymm6, ymm3
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm1, dword ptr [rax + 24]
    vfmadd231ps ymm4, ymm0, ymm1
    # sum of group 7
    vpmaddwd ymm0, ymm7, ymm3
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm1, dword ptr [rax + 28]
    vfmadd231ps ymm4, ymm0, ymm1
    # the lanes summed
    vxorps xmm0, xmm0, xmm0
    vaddps ymm4, ymm4, ymm0
    vextractf128 xmm0, ymm4, 1
    vaddps xmm0, xmm4, xmm0
    vmovhlps xmm1, xmm0, xmm0
    vaddps xmm0, xmm0, xmm1
    vmovshdup xmm1, xmm0
    vaddss xmm0, xmm0, xmm1
    mov rsp, rbp
    pop rbp
    pop r12
    pop rbx
    vzeroupper
    ret
4:
    vxorps xmm0, xmm0, xmm0
    mov rsp, rbp
    pop rbp
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
#   01 01 01 01
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
