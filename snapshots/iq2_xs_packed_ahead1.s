# iq2_xs_packed_ahead1(gpr, gpr, gpr, gpr, gpr) -> vec
# frame: 256 bytes aligned to 256, saves [rbx, r12]
# row=rdi sbs=rsi quants=rdx scales=rcx grid=r8 spread=ymm0
# bit=ymm1 max8=ymm2 one16=ymm3 acc=ymm4 cur=rax nxt=r9
# p0=ymm5 p1=ymm7 p2=ymm5 p3=ymm6 p4=ymm5 p5=ymm6
# p6=ymm5 p7=ymm6 packed_signs=ymm5,ymm6 d=ymm5 steps=ymm6,ymm7 low_scales=ymm7,ymm8
# high_scales=ymm6,ymm7 index0=r11,rsi index1=rbx,r9,rdi index2=r12,r10,r9 index3=r10,rcx packed_indices=r10,rcx
# rows=ymm6,ymm8,ymm7 signs=ymm6,ymm8,ymm7 x=ymm7,ymm9,ymm8,ymm0 lanes=ymm5,ymm6,ymm0 low_scale=ymm8,ymm7,ymm6,ymm1 high_scale=ymm9,ymm8,ymm2
# scale=ymm8,ymm7,ymm6,ymm1 zero=ymm0 high=ymm0 sum=ymm0 pair=ymm1 result=ymm0
# nothing=ymm0
    push rbx
    push r12
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
    lea rax, [rsp]
    lea r9, [rsp + 128]
    # signs of superblock +0
    vmovdqu ymm5, ymmword ptr [rdi + 38]
    vmovdqa ymmword ptr [rax + 64], ymm5
    # scales of superblock +0
    vpbroadcastw xmm5, word ptr [rdi]
    vcvtph2ps ymm5, xmm5
    vpmovzxbd ymm6, qword ptr [rdi + 70]
    vpand ymm7, ymm6, ymmword ptr [rip + {c4}]
    vpsrld ymm6, ymm6, 4
    vpaddd ymm7, ymm7, ymm7
    vpor ymm7, ymm7, ymmword ptr [rip + {c5}]
    vcvtdq2ps ymm7, ymm7
    vmulps ymm7, ymm5, ymm7
    vmulps ymm7, ymm7, ymmword ptr [rip + {c6}]
    vmulps ymm7, ymm7, ymmword ptr [rcx]
    vpaddd ymm6, ymm6, ymm6
    vpor ymm6, ymm6, ymmword ptr [rip + {c5}]
    vcvtdq2ps ymm6, ymm6
    vmulps ymm6, ymm5, ymm6
    vmulps ymm6, ymm6, ymmword ptr [rip + {c6}]
    vmulps ymm6, ymm6, ymmword ptr [rcx]
    vmovaps ymmword ptr [rax], ymm7
    vmovaps ymmword ptr [rax + 32], ymm6
    # products of group 0 of superblock +0
    mov r10, qword ptr [rdi + 2]
    mov r11, r10
    and r11, 511
    mov rbx, r10
    shr rbx, 9
    and rbx, 511
    mov r12, r10
    shr r12, 18
    and r12, 511
    mov r10, r10
    shr r10, 27
    and r10, 511
    vmovq xmm5, qword ptr [r8 + r11*8]
    vmovhps xmm5, xmm5, qword ptr [r8 + rbx*8]
    vmovq xmm6, qword ptr [r8 + r12*8]
    vmovhps xmm6, xmm6, qword ptr [r8 + r10*8]
    vinserti128 ymm5, ymm5, xmm6, 1
    vpbroadcastd ymm6, dword ptr [rax + 64]
    vpshufb ymm6, ymm6, ymm0
    vpand ymm6, ymm6, ymm1
    vpaddb ymm6, ymm6, ymm2
    vmovdqa ymm7, ymmword ptr [rdx]
    vpsignb ymm7, ymm7, ymm6
    vpmaddubsw ymm5, ymm5, ymm7
    dec rsi
    jz 3f
2:
    # signs of superblock +1
    vmovdqu ymm6, ymmword ptr [rdi + 116]
    # products of group 1 of superblock +0
    mov r10, qword ptr [rdi + 6]
    shr r10, 4
    mov r11, r10
    and r11, 511
    mov rbx, r10
    shr rbx, 9
    and rbx, 511
    mov r12, r10
    shr r12, 18
    and r12, 511
    mov r10, r10
    shr r10, 27
    and r10, 511
    vmovq xmm7, qword ptr [r8 + r11*8]
    vmovhps xmm7, xmm7, qword ptr [r8 + rbx*8]
    vmovq xmm8, qword ptr [r8 + r12*8]
    vmovhps xmm8, xmm8, qword ptr [r8 + r10*8]
    vinserti128 ymm7, ymm7, xmm8, 1
    vpbroadcastd ymm8, dword ptr [rax + 68]
    vpshufb ymm8, ymm8, ymm0
    vpand ymm8, ymm8, ymm1
    vpaddb ymm8, ymm8, ymm2
    vmovdqa ymm9, ymmword ptr [rdx + 32]
    vpsignb ymm9, ymm9, ymm8
    vpmaddubsw ymm7, ymm7, ymm9
    # sum of group 0
    vpmaddwd ymm5, ymm5, ymm3
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm8, dword ptr [rax]
    vbroadcastss ymm9, dword ptr [rax + 32]
    vpblendd ymm8, ymm8, ymm9, 0xf0
    vfmadd231ps ymm4, ymm5, ymm8
    vmovdqa ymmword ptr [r9 + 64], ymm6
    # products of group 2 of superblock +0
    mov r10, qword ptr [rdi + 11]
    mov r11, r10
    and r11, 511
    mov rbx, r10
    shr rbx, 9
    and rbx, 511
    mov r12, r10
    shr r12, 18
    and r12, 511
    mov r10, r10
    shr r10, 27
    and r10, 511
    vmovq xmm5, qword ptr [r8 + r11*8]
    vmovhps xmm5, xmm5, qword ptr [r8 + rbx*8]
    vmovq xmm6, qword ptr [r8 + r12*8]
    vmovhps xmm6, xmm6, qword ptr [r8 + r10*8]
    vinserti128 ymm5, ymm5, xmm6, 1
    vpbroadcastd ymm6, dword ptr [rax + 72]
    vpshufb ymm6, ymm6, ymm0
    vpand ymm6, ymm6, ymm1
    vpaddb ymm6, ymm6, ymm2
    vmovdqa ymm8, ymmword ptr [rdx + 64]
    vpsignb ymm8, ymm8, ymm6
    vpmaddubsw ymm5, ymm5, ymm8
    # sum of group 1
    vpmaddwd ymm6, ymm7, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 4]
    vbroadcastss ymm8, dword ptr [rax + 36]
    vpblendd ymm7, ymm7, ymm8, 0xf0
    vfmadd231ps ymm4, ymm6, ymm7
    # products of group 3 of superblock +0
    mov r10, qword ptr [rdi + 15]
    shr r10, 4
    mov r11, r10
    and r11, 511
    mov rbx, r10
    shr rbx, 9
    and rbx, 511
    mov r12, r10
    shr r12, 18
    and r12, 511
    mov r10, r10
    shr r10, 27
    and r10, 511
    vmovq xmm6, qword ptr [r8 + r11*8]
    vmovhps xmm6, xmm6, qword ptr [r8 + rbx*8]
    vmovq xmm7, qword ptr [r8 + r12*8]
    vmovhps xmm7, xmm7, qword ptr [r8 + r10*8]
    vinserti128 ymm6, ymm6, xmm7, 1
    vpbroadcastd ymm7, dword ptr [rax + 76]
    vpshufb ymm7, ymm7, ymm0
    vpand ymm7, ymm7, ymm1
    vpaddb ymm7, ymm7, ymm2
    vmovdqa ymm8, ymmword ptr [rdx + 96]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm6, ymm6, ymm8
    # sum of group 2
    vpmaddwd ymm5, ymm5, ymm3
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm7, dword ptr [rax + 8]
    vbroadcastss ymm8, dword ptr [rax + 40]
    vpblendd ymm7, ymm7, ymm8, 0xf0
    vfmadd231ps ymm4, ymm5, ymm7
    # products of group 4 of superblock +0
    mov r10, qword ptr [rdi + 20]
    mov r11, r10
    and r11, 511
    mov rbx, r10
    shr rbx, 9
    and rbx, 511
    mov r12, r10
    shr r12, 18
    and r12, 511
    mov r10, r10
    shr r10, 27
    and r10, 511
    vmovq xmm5, qword ptr [r8 + r11*8]
    vmovhps xmm5, xmm5, qword ptr [r8 + rbx*8]
    vmovq xmm7, qword ptr [r8 + r12*8]
    vmovhps xmm7, xmm7, qword ptr [r8 + r10*8]
    vinserti128 ymm5, ymm5, xmm7, 1
    vpbroadcastd ymm7, dword ptr [rax + 80]
    vpshufb ymm7, ymm7, ymm0
    vpand ymm7, ymm7, ymm1
    vpaddb ymm7, ymm7, ymm2
    vmovdqa ymm8, ymmword ptr [rdx + 128]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm5, ymm5, ymm8
    # sum of group 3
    vpmaddwd ymm6, ymm6, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 12]
    vbroadcastss ymm8, dword ptr [rax + 44]
    vpblendd ymm7, ymm7, ymm8, 0xf0
    vfmadd231ps ymm4, ymm6, ymm7
    # products of group 5 of superblock +0
    mov r10, qword ptr [rdi + 24]
    shr r10, 4
    mov r11, r10
    and r11, 511
    mov rbx, r10
    shr rbx, 9
    and rbx, 511
    mov r12, r10
    shr r12, 18
    and r12, 511
    mov r10, r10
    shr r10, 27
    and r10, 511
    vmovq xmm6, qword ptr [r8 + r11*8]
    vmovhps xmm6, xmm6, qword ptr [r8 + rbx*8]
    vmovq xmm7, qword ptr [r8 + r12*8]
    vmovhps xmm7, xmm7, qword ptr [r8 + r10*8]
    vinserti128 ymm6, ymm6, xmm7, 1
    vpbroadcastd ymm7, dword ptr [rax + 84]
    vpshufb ymm7, ymm7, ymm0
    vpand ymm7, ymm7, ymm1
    vpaddb ymm7, ymm7, ymm2
    vmovdqa ymm8, ymmword ptr [rdx + 160]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm6, ymm6, ymm8
    # sum of group 4
    vpmaddwd ymm5, ymm5, ymm3
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm7, dword ptr [rax + 16]
    vbroadcastss ymm8, dword ptr [rax + 48]
    vpblendd ymm7, ymm7, ymm8, 0xf0
    vfmadd231ps ymm4, ymm5, ymm7
    # products of group 6 of superblock +0
    mov r10, qword ptr [rdi + 29]
    mov r11, r10
    and r11, 511
    mov rbx, r10
    shr rbx, 9
    and rbx, 511
    mov r12, r10
    shr r12, 18
    and r12, 511
    mov r10, r10
    shr r10, 27
    and r10, 511
    vmovq xmm5, qword ptr [r8 + r11*8]
    vmovhps xmm5, xmm5, qword ptr [r8 + rbx*8]
    vmovq xmm7, qword ptr [r8 + r12*8]
    vmovhps xmm7, xmm7, qword ptr [r8 + r10*8]
    vinserti128 ymm5, ymm5, xmm7, 1
    vpbroadcastd ymm7, dword ptr [rax + 88]
    vpshufb ymm7, ymm7, ymm0
    vpand ymm7, ymm7, ymm1
    vpaddb ymm7, ymm7, ymm2
    vmovdqa ymm8, ymmword ptr [rdx + 192]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm5, ymm5, ymm8
    # sum of group 5
    vpmaddwd ymm6, ymm6, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 20]
    vbroadcastss ymm8, dword ptr [rax + 52]
    vpblendd ymm7, ymm7, ymm8, 0xf0
    vfmadd231ps ymm4, ymm6, ymm7
    # products of group 7 of superblock +0
    mov r10, qword ptr [rdi + 33]
    shr r10, 4
    mov r11, r10
    and r11, 511
    mov rbx, r10
    shr rbx, 9
    and rbx, 511
    mov r12, r10
    shr r12, 18
    and r12, 511
    mov r10, r10
    shr r10, 27
    and r10, 511
    vmovq xmm6, qword ptr [r8 + r11*8]
    vmovhps xmm6, xmm6, qword ptr [r8 + rbx*8]
    vmovq xmm7, qword ptr [r8 + r12*8]
    vmovhps xmm7, xmm7, qword ptr [r8 + r10*8]
    vinserti128 ymm6, ymm6, xmm7, 1
    vpbroadcastd ymm7, dword ptr [rax + 92]
    vpshufb ymm7, ymm7, ymm0
    vpand ymm7, ymm7, ymm1
    vpaddb ymm7, ymm7, ymm2
    vmovdqa ymm8, ymmword ptr [rdx + 224]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm6, ymm6, ymm8
    # sum of group 6
    vpmaddwd ymm5, ymm5, ymm3
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm7, dword ptr [rax + 24]
    vbroadcastss ymm8, dword ptr [rax + 56]
    vpblendd ymm7, ymm7, ymm8, 0xf0
    vfmadd231ps ymm4, ymm5, ymm7
    # scales of superblock +1
    vpbroadcastw xmm5, word ptr [rdi + 78]
    vcvtph2ps ymm5, xmm5
    vpmovzxbd ymm7, qword ptr [rdi + 148]
    vpand ymm8, ymm7, ymmword ptr [rip + {c4}]
    vpsrld ymm7, ymm7, 4
    vpaddd ymm8, ymm8, ymm8
    vpor ymm8, ymm8, ymmword ptr [rip + {c5}]
    vcvtdq2ps ymm8, ymm8
    vmulps ymm8, ymm5, ymm8
    vmulps ymm8, ymm8, ymmword ptr [rip + {c6}]
    vmulps ymm8, ymm8, ymmword ptr [rcx + 32]
    vpaddd ymm7, ymm7, ymm7
    vpor ymm7, ymm7, ymmword ptr [rip + {c5}]
    vcvtdq2ps ymm7, ymm7
    vmulps ymm7, ymm5, ymm7
    vmulps ymm7, ymm7, ymmword ptr [rip + {c6}]
    vmulps ymm7, ymm7, ymmword ptr [rcx + 32]
    vmovaps ymmword ptr [r9], ymm8
    vmovaps ymmword ptr [r9 + 32], ymm7
    # products of group 0 of superblock +1
    mov r10, qword ptr [rdi + 80]
    mov r11, r10
    and r11, 511
    mov rbx, r10
    shr rbx, 9
    and rbx, 511
    mov r12, r10
    shr r12, 18
    and r12, 511
    mov r10, r10
    shr r10, 27
    and r10, 511
    vmovq xmm5, qword ptr [r8 + r11*8]
    vmovhps xmm5, xmm5, qword ptr [r8 + rbx*8]
    vmovq xmm7, qword ptr [r8 + r12*8]
    vmovhps xmm7, xmm7, qword ptr [r8 + r10*8]
    vinserti128 ymm5, ymm5, xmm7, 1
    vpbroadcastd ymm7, dword ptr [r9 + 64]
    vpshufb ymm7, ymm7, ymm0
    vpand ymm7, ymm7, ymm1
    vpaddb ymm7, ymm7, ymm2
    vmovdqa ymm8, ymmword ptr [rdx + 256]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm5, ymm5, ymm8
    # sum of group 7
    vpmaddwd ymm6, ymm6, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 28]
    vbroadcastss ymm8, dword ptr [rax + 60]
    vpblendd ymm7, ymm7, ymm8, 0xf0
    vfmadd231ps ymm4, ymm6, ymm7
    add rdi, 78
    add rdx, 256
    add rcx, 32
    xor rax, 128
    xor r9, 128
    dec rsi
    jnz 2b
3:
    # products of group 1 of superblock +0
    mov rcx, qword ptr [rdi + 6]
    shr rcx, 4
    mov rsi, rcx
    and rsi, 511
    mov r9, rcx
    shr r9, 9
    and r9, 511
    mov r10, rcx
    shr r10, 18
    and r10, 511
    mov rcx, rcx
    shr rcx, 27
    and rcx, 511
    vmovq xmm7, qword ptr [r8 + rsi*8]
    vmovhps xmm7, xmm7, qword ptr [r8 + r9*8]
    vmovq xmm6, qword ptr [r8 + r10*8]
    vmovhps xmm6, xmm6, qword ptr [r8 + rcx*8]
    vinserti128 ymm7, ymm7, xmm6, 1
    vpbroadcastd ymm6, dword ptr [rax + 68]
    vpshufb ymm6, ymm6, ymm0
    vpand ymm6, ymm6, ymm1
    vpaddb ymm6, ymm6, ymm2
    vmovdqa ymm8, ymmword ptr [rdx + 32]
    vpsignb ymm8, ymm8, ymm6
    vpmaddubsw ymm7, ymm7, ymm8
    # sum of group 0
    vpmaddwd ymm5, ymm5, ymm3
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm6, dword ptr [rax]
    vbroadcastss ymm8, dword ptr [rax + 32]
    vpblendd ymm6, ymm6, ymm8, 0xf0
    vfmadd231ps ymm4, ymm5, ymm6
    # products of group 2 of superblock +0
    mov rcx, qword ptr [rdi + 11]
    mov rsi, rcx
    and rsi, 511
    mov r9, rcx
    shr r9, 9
    and r9, 511
    mov r10, rcx
    shr r10, 18
    and r10, 511
    mov rcx, rcx
    shr rcx, 27
    and rcx, 511
    vmovq xmm5, qword ptr [r8 + rsi*8]
    vmovhps xmm5, xmm5, qword ptr [r8 + r9*8]
    vmovq xmm6, qword ptr [r8 + r10*8]
    vmovhps xmm6, xmm6, qword ptr [r8 + rcx*8]
    vinserti128 ymm5, ymm5, xmm6, 1
    vpbroadcastd ymm6, dword ptr [rax + 72]
    vpshufb ymm6, ymm6, ymm0
    vpand ymm6, ymm6, ymm1
    vpaddb ymm6, ymm6, ymm2
    vmovdqa ymm8, ymmword ptr [rdx + 64]
    vpsignb ymm8, ymm8, ymm6
    vpmaddubsw ymm5, ymm5, ymm8
    # sum of group 1
    vpmaddwd ymm6, ymm7, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 4]
    vbroadcastss ymm8, dword ptr [rax + 36]
    vpblendd ymm7, ymm7, ymm8, 0xf0
    vfmadd231ps ymm4, ymm6, ymm7
    # products of group 3 of superblock +0
    mov rcx, qword ptr [rdi + 15]
    shr rcx, 4
    mov rsi, rcx
    and rsi, 511
    mov r9, rcx
    shr r9, 9
    and r9, 511
    mov r10, rcx
    shr r10, 18
    and r10, 511
    mov rcx, rcx
    shr rcx, 27
    and rcx, 511
    vmovq xmm6, qword ptr [r8 + rsi*8]
    vmovhps xmm6, xmm6, qword ptr [r8 + r9*8]
    vmovq xmm7, qword ptr [r8 + r10*8]
    vmovhps xmm7, xmm7, qword ptr [r8 + rcx*8]
    vinserti128 ymm6, ymm6, xmm7, 1
    vpbroadcastd ymm7, dword ptr [rax + 76]
    vpshufb ymm7, ymm7, ymm0
    vpand ymm7, ymm7, ymm1
    vpaddb ymm7, ymm7, ymm2
    vmovdqa ymm8, ymmword ptr [rdx + 96]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm6, ymm6, ymm8
    # sum of group 2
    vpmaddwd ymm5, ymm5, ymm3
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm7, dword ptr [rax + 8]
    vbroadcastss ymm8, dword ptr [rax + 40]
    vpblendd ymm7, ymm7, ymm8, 0xf0
    vfmadd231ps ymm4, ymm5, ymm7
    # products of group 4 of superblock +0
    mov rcx, qword ptr [rdi + 20]
    mov rsi, rcx
    and rsi, 511
    mov r9, rcx
    shr r9, 9
    and r9, 511
    mov r10, rcx
    shr r10, 18
    and r10, 511
    mov rcx, rcx
    shr rcx, 27
    and rcx, 511
    vmovq xmm5, qword ptr [r8 + rsi*8]
    vmovhps xmm5, xmm5, qword ptr [r8 + r9*8]
    vmovq xmm7, qword ptr [r8 + r10*8]
    vmovhps xmm7, xmm7, qword ptr [r8 + rcx*8]
    vinserti128 ymm5, ymm5, xmm7, 1
    vpbroadcastd ymm7, dword ptr [rax + 80]
    vpshufb ymm7, ymm7, ymm0
    vpand ymm7, ymm7, ymm1
    vpaddb ymm7, ymm7, ymm2
    vmovdqa ymm8, ymmword ptr [rdx + 128]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm5, ymm5, ymm8
    # sum of group 3
    vpmaddwd ymm6, ymm6, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 12]
    vbroadcastss ymm8, dword ptr [rax + 44]
    vpblendd ymm7, ymm7, ymm8, 0xf0
    vfmadd231ps ymm4, ymm6, ymm7
    # products of group 5 of superblock +0
    mov rcx, qword ptr [rdi + 24]
    shr rcx, 4
    mov rsi, rcx
    and rsi, 511
    mov r9, rcx
    shr r9, 9
    and r9, 511
    mov r10, rcx
    shr r10, 18
    and r10, 511
    mov rcx, rcx
    shr rcx, 27
    and rcx, 511
    vmovq xmm6, qword ptr [r8 + rsi*8]
    vmovhps xmm6, xmm6, qword ptr [r8 + r9*8]
    vmovq xmm7, qword ptr [r8 + r10*8]
    vmovhps xmm7, xmm7, qword ptr [r8 + rcx*8]
    vinserti128 ymm6, ymm6, xmm7, 1
    vpbroadcastd ymm7, dword ptr [rax + 84]
    vpshufb ymm7, ymm7, ymm0
    vpand ymm7, ymm7, ymm1
    vpaddb ymm7, ymm7, ymm2
    vmovdqa ymm8, ymmword ptr [rdx + 160]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm6, ymm6, ymm8
    # sum of group 4
    vpmaddwd ymm5, ymm5, ymm3
    vcvtdq2ps ymm5, ymm5
    vbroadcastss ymm7, dword ptr [rax + 16]
    vbroadcastss ymm8, dword ptr [rax + 48]
    vpblendd ymm7, ymm7, ymm8, 0xf0
    vfmadd231ps ymm4, ymm5, ymm7
    # products of group 6 of superblock +0
    mov rcx, qword ptr [rdi + 29]
    mov rsi, rcx
    and rsi, 511
    mov r9, rcx
    shr r9, 9
    and r9, 511
    mov r10, rcx
    shr r10, 18
    and r10, 511
    mov rcx, rcx
    shr rcx, 27
    and rcx, 511
    vmovq xmm5, qword ptr [r8 + rsi*8]
    vmovhps xmm5, xmm5, qword ptr [r8 + r9*8]
    vmovq xmm7, qword ptr [r8 + r10*8]
    vmovhps xmm7, xmm7, qword ptr [r8 + rcx*8]
    vinserti128 ymm5, ymm5, xmm7, 1
    vpbroadcastd ymm7, dword ptr [rax + 88]
    vpshufb ymm7, ymm7, ymm0
    vpand ymm7, ymm7, ymm1
    vpaddb ymm7, ymm7, ymm2
    vmovdqa ymm8, ymmword ptr [rdx + 192]
    vpsignb ymm8, ymm8, ymm7
    vpmaddubsw ymm5, ymm5, ymm8
    # sum of group 5
    vpmaddwd ymm6, ymm6, ymm3
    vcvtdq2ps ymm6, ymm6
    vbroadcastss ymm7, dword ptr [rax + 20]
    vbroadcastss ymm8, dword ptr [rax + 52]
    vpblendd ymm7, ymm7, ymm8, 0xf0
    vfmadd231ps ymm4, ymm6, ymm7
    # products of group 7 of superblock +0
    mov rcx, qword ptr [rdi + 33]
    shr rcx, 4
    mov rsi, rcx
    and rsi, 511
    mov rdi, rcx
    shr rdi, 9
    and rdi, 511
    mov r9, rcx
    shr r9, 18
    and r9, 511
    mov rcx, rcx
    shr rcx, 27
    and rcx, 511
    vmovq xmm6, qword ptr [r8 + rsi*8]
    vmovhps xmm6, xmm6, qword ptr [r8 + rdi*8]
    vmovq xmm7, qword ptr [r8 + r9*8]
    vmovhps xmm7, xmm7, qword ptr [r8 + rcx*8]
    vinserti128 ymm6, ymm6, xmm7, 1
    vpbroadcastd ymm7, dword ptr [rax + 92]
    vpshufb ymm7, ymm7, ymm0
    vpand ymm7, ymm7, ymm1
    vpaddb ymm7, ymm7, ymm2
    vmovdqa ymm0, ymmword ptr [rdx + 224]
    vpsignb ymm0, ymm0, ymm7
    vpmaddubsw ymm6, ymm6, ymm0
    # sum of group 6
    vpmaddwd ymm0, ymm5, ymm3
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm1, dword ptr [rax + 24]
    vbroadcastss ymm2, dword ptr [rax + 56]
    vpblendd ymm1, ymm1, ymm2, 0xf0
    vfmadd231ps ymm4, ymm0, ymm1
    # sum of group 7
    vpmaddwd ymm0, ymm6, ymm3
    vcvtdq2ps ymm0, ymm0
    vbroadcastss ymm1, dword ptr [rax + 28]
    vbroadcastss ymm2, dword ptr [rax + 60]
    vpblendd ymm1, ymm1, ymm2, 0xf0
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
