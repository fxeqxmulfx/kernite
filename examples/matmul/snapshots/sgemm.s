# sgemm(gpr, gpr, gpr, gpr, gpr, gpr) -> ()
# needs: avx, avx2, fma
# frame: 88 bytes aligned to 8, saves [rbx, r12, r13, r14, r15]
# c, spilled: 8 bytes at rsp
# n, spilled: 8 bytes at rsp + 8
# k, spilled: 8 bytes at rsp + 16
# ak, spilled: 8 bytes at rsp + 24
# bk, spilled: 8 bytes at rsp + 32
# m, spilled: 8 bytes at rsp + 40
# ap, spilled: 8 bytes at rsp + 48
# bj, spilled: 8 bytes at rsp + 56
# cj, spilled: 8 bytes at rsp + 64
# nleft, spilled: 8 bytes at rsp + 72
# bp, spilled: 8 bytes at rsp + 80
# c=rdi,rsi a=rsi b=rdx shape=rcx ap=r8,rax,rdi bp=r9,rax,rdi,r10
# ld=rax pc=rcx kc=rdx most=rsi,rdi,r11,r15 lda=rsi q=rdi
# p=r8,r10,r13 left=r9 s=r10,ymm14 s3=r11 l=rbx,r11,r13 f=r12
# r=r10 at=rdi from_row=r12 d=r11 n=rbx,rax,rsi g=r13
# nc=rsi col=r8 mask0=ymm1,ymm12,ymm13 mask1=ymm0,ymm13,ymm12 t=r10,r14,rdx lanes=ymm0,ymm12
# v=ymm2 pa=rdi ci=r8 hleft=r9 pb=r10 ct=r11
# wleft=rbx x=r12 acc0=ymm0 acc1=ymm1 acc2=ymm2 acc3=ymm3
# acc4=ymm4 acc5=ymm5 acc6=ymm6 acc7=ymm7 acc8=ymm8 acc9=ymm9
# acc10=ymm10 acc11=ymm11 b0=ymm12 b1=ymm13 k=rax,rdx,rsi ak=rsi,r8,rdx
# bk=rdx,rsi m=rax,r9 bj=rsi,r8 cj=rsi,r8 nleft=rsi
    push rbx
    push r12
    push r13
    push r14
    push r15
    push rbp
    mov rbp, rsp
    sub rsp, 88
    # bp spilled
    mov qword ptr [rsp + 80], r9
    # ap spilled
    mov qword ptr [rsp + 48], r8
    # c spilled
    mov qword ptr [rsp], rdi
    mov rax, qword ptr [rcx]
    # m spilled
    mov qword ptr [rsp + 40], rax
    mov rax, qword ptr [rcx + 8]
    # n spilled
    mov qword ptr [rsp + 8], rax
    mov rax, qword ptr [rcx + 16]
    # k spilled
    mov qword ptr [rsp + 16], rax
    # m reloaded
    mov rax, qword ptr [rsp + 40]
    test rax, rax
    jz 28f
    # n reloaded
    mov rax, qword ptr [rsp + 8]
    test rax, rax
    jz 28f
    # ap reloaded
    mov rax, qword ptr [rsp + 48]
    add rax, 63
    # ap spilled
    mov qword ptr [rsp + 48], rax
    # ap reloaded
    mov rax, qword ptr [rsp + 48]
    and rax, -64
    # ap spilled
    mov qword ptr [rsp + 48], rax
    # bp reloaded
    mov rax, qword ptr [rsp + 80]
    add rax, 63
    # bp spilled
    mov qword ptr [rsp + 80], rax
    # bp reloaded
    mov rax, qword ptr [rsp + 80]
    and rax, -64
    # bp spilled
    mov qword ptr [rsp + 80], rax
    # the bytes from a row of b or c to the next
    # n reloaded
    mov rax, qword ptr [rsp + 8]
    shl rax, 2
    # blocks of KC steps of k from step pc, each chain going on from the block before
    xor ecx, ecx
    # ak spilled
    mov qword ptr [rsp + 24], rsi
    # bk spilled
    mov qword ptr [rsp + 32], rdx
2:
    # k reloaded
    mov rdx, qword ptr [rsp + 16]
    sub rdx, rcx
    mov rsi, 384
    cmp rdx, rsi
    cmova rdx, rsi
    test rdx, rdx
    jz 12f
    # k reloaded
    mov rsi, qword ptr [rsp + 16]
    shl rsi, 2
    # ap reloaded
    mov rdi, qword ptr [rsp + 48]
    # ak reloaded
    mov r8, qword ptr [rsp + 24]
    # m reloaded
    mov r9, qword ptr [rsp + 40]
3:
    cmp r9, 6
    jb 5f
    # a whole sliver, a step at a time: rows 0 to 2 from s, and 3 to 5 from s3
    mov r10, r8
    lea r11, [r8 + rsi*2]
    add r11, rsi
    mov rbx, rdx
4:
    mov r12d, dword ptr [r10]
    mov dword ptr [rdi], r12d
    mov r12d, dword ptr [r10 + rsi*1]
    mov dword ptr [rdi + 4], r12d
    mov r12d, dword ptr [r10 + rsi*2]
    mov dword ptr [rdi + 8], r12d
    mov r12d, dword ptr [r11]
    mov dword ptr [rdi + 12], r12d
    mov r12d, dword ptr [r11 + rsi*1]
    mov dword ptr [rdi + 16], r12d
    mov r12d, dword ptr [r11 + rsi*2]
    mov dword ptr [rdi + 20], r12d
    add r10, 4
    add r11, 4
    add rdi, 24
    dec rbx
    jnz 4b
    lea r8, [r8 + rsi*4]
    lea r8, [r8 + rsi*2]
    sub r9, 6
    ja 3b
    jmp 12f
    # the last sliver, short of MR rows: a row at a time, then zeros
5:
    xor r10d, r10d
6:
    mov r11, rdi
    mov rbx, rdx
    cmp r10, r9
    jae 8f
    mov r12, r8
7:
    mov r13d, dword ptr [r12]
    mov dword ptr [r11], r13d
    add r12, 4
    add r11, 24
    dec rbx
    jnz 7b
    jmp 9f
8:
    mov dword ptr [r11], 0
    add r11, 24
    dec rbx
    jnz 8b
9:
    add r8, rsi
    add rdi, 4
    inc r10
    cmp r10, 6
    jb 6b
12:
    # blocks of NC columns of b, and of c
    # bk reloaded
    mov rsi, qword ptr [rsp + 32]
    # bj spilled
    mov qword ptr [rsp + 56], rsi
    # c reloaded
    mov rsi, qword ptr [rsp]
    # cj spilled
    mov qword ptr [rsp + 64], rsi
    # n reloaded
    mov rsi, qword ptr [rsp + 8]
    # nleft spilled
    mov qword ptr [rsp + 72], rsi
13:
    # nleft reloaded
    mov rsi, qword ptr [rsp + 72]
    mov rdi, 256
    cmp rsi, rdi
    cmova rsi, rdi
    test rdx, rdx
    jz 16f
    # bp reloaded
    mov rdi, qword ptr [rsp + 80]
    # bj reloaded
    mov r8, qword ptr [rsp + 56]
    mov r9, rsi
14:
    mov r10, r9
    mov r11, 16
    cmp r10, r11
    cmova r10, r11
    vmovd xmm0, r10d
    vpbroadcastd ymm0, xmm0
    vpcmpgtd ymm1, ymm0, ymmword ptr [rip + {c0}]
    vpcmpgtd ymm0, ymm0, ymmword ptr [rip + {c1}]
    mov r10, r8
    mov r11, rdx
15:
    vmaskmovps ymm2, ymm1, ymmword ptr [r10]
    vmovaps ymmword ptr [rdi], ymm2
    vmaskmovps ymm2, ymm0, ymmword ptr [r10 + 32]
    vmovaps ymmword ptr [rdi + 32], ymm2
    add r10, rax
    add rdi, 64
    dec r11
    jnz 15b
    add r8, 64
    sub r9, 16
    ja 14b
16:
    # slivers of MR rows of a, and of c
    # ap reloaded
    mov rdi, qword ptr [rsp + 48]
    # cj reloaded
    mov r8, qword ptr [rsp + 64]
    # m reloaded
    mov r9, qword ptr [rsp + 40]
17:
    # slivers of NR columns of the block, a tile of c each
    # bp reloaded
    mov r10, qword ptr [rsp + 80]
    mov r11, r8
    mov rbx, rsi
18:
    mov r12, rdi
    vxorps xmm0, xmm0, xmm0
    vxorps xmm1, xmm1, xmm1
    vxorps xmm2, xmm2, xmm2
    vxorps xmm3, xmm3, xmm3
    vxorps xmm4, xmm4, xmm4
    vxorps xmm5, xmm5, xmm5
    vxorps xmm6, xmm6, xmm6
    vxorps xmm7, xmm7, xmm7
    vxorps xmm8, xmm8, xmm8
    vxorps xmm9, xmm9, xmm9
    vxorps xmm10, xmm10, xmm10
    vxorps xmm11, xmm11, xmm11
    test rcx, rcx
    jz 21f
    mov r13, r11
    cmp r9, 6
    jb 19f
    cmp rbx, 16
    jb 19f
    vmovups ymm0, ymmword ptr [r13]
    vmovups ymm1, ymmword ptr [r13 + 32]
    add r13, rax
    vmovups ymm2, ymmword ptr [r13]
    vmovups ymm3, ymmword ptr [r13 + 32]
    add r13, rax
    vmovups ymm4, ymmword ptr [r13]
    vmovups ymm5, ymmword ptr [r13 + 32]
    add r13, rax
    vmovups ymm6, ymmword ptr [r13]
    vmovups ymm7, ymmword ptr [r13 + 32]
    add r13, rax
    vmovups ymm8, ymmword ptr [r13]
    vmovups ymm9, ymmword ptr [r13 + 32]
    add r13, rax
    vmovups ymm10, ymmword ptr [r13]
    vmovups ymm11, ymmword ptr [r13 + 32]
    add r13, rax
    jmp 20f
19:
    mov r14, rbx
    mov r15, 16
    cmp r14, r15
    cmova r14, r15
    vmovd xmm0, r14d
    vpbroadcastd ymm0, xmm0
    vpcmpgtd ymm12, ymm0, ymmword ptr [rip + {c0}]
    vpcmpgtd ymm13, ymm0, ymmword ptr [rip + {c1}]
    vmaskmovps ymm0, ymm12, ymmword ptr [r13]
    vmaskmovps ymm1, ymm13, ymmword ptr [r13 + 32]
    add r13, rax
    cmp r9, 1
    jbe 20f
    vmaskmovps ymm2, ymm12, ymmword ptr [r13]
    vmaskmovps ymm3, ymm13, ymmword ptr [r13 + 32]
    add r13, rax
    cmp r9, 2
    jbe 20f
    vmaskmovps ymm4, ymm12, ymmword ptr [r13]
    vmaskmovps ymm5, ymm13, ymmword ptr [r13 + 32]
    add r13, rax
    cmp r9, 3
    jbe 20f
    vmaskmovps ymm6, ymm12, ymmword ptr [r13]
    vmaskmovps ymm7, ymm13, ymmword ptr [r13 + 32]
    add r13, rax
    cmp r9, 4
    jbe 20f
    vmaskmovps ymm8, ymm12, ymmword ptr [r13]
    vmaskmovps ymm9, ymm13, ymmword ptr [r13 + 32]
    add r13, rax
    cmp r9, 5
    jbe 20f
    vmaskmovps ymm10, ymm12, ymmword ptr [r13]
    vmaskmovps ymm11, ymm13, ymmword ptr [r13 + 32]
    add r13, rax
20:
21:
    # UNROLL steps of k a turn
    mov r13, rdx
    sub r13, 4
    jb 23f
22:
    vmovaps ymm12, ymmword ptr [r10]
    vmovaps ymm13, ymmword ptr [r10 + 32]
    vbroadcastss ymm14, dword ptr [r12]
    vfmadd231ps ymm0, ymm14, ymm12
    vfmadd231ps ymm1, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r12 + 4]
    vfmadd231ps ymm2, ymm14, ymm12
    vfmadd231ps ymm3, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r12 + 8]
    vfmadd231ps ymm4, ymm14, ymm12
    vfmadd231ps ymm5, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r12 + 12]
    vfmadd231ps ymm6, ymm14, ymm12
    vfmadd231ps ymm7, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r12 + 16]
    vfmadd231ps ymm8, ymm14, ymm12
    vfmadd231ps ymm9, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r12 + 20]
    vfmadd231ps ymm10, ymm14, ymm12
    vfmadd231ps ymm11, ymm14, ymm13
    vmovaps ymm12, ymmword ptr [r10 + 64]
    vmovaps ymm13, ymmword ptr [r10 + 96]
    vbroadcastss ymm14, dword ptr [r12 + 24]
    vfmadd231ps ymm0, ymm14, ymm12
    vfmadd231ps ymm1, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r12 + 28]
    vfmadd231ps ymm2, ymm14, ymm12
    vfmadd231ps ymm3, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r12 + 32]
    vfmadd231ps ymm4, ymm14, ymm12
    vfmadd231ps ymm5, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r12 + 36]
    vfmadd231ps ymm6, ymm14, ymm12
    vfmadd231ps ymm7, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r12 + 40]
    vfmadd231ps ymm8, ymm14, ymm12
    vfmadd231ps ymm9, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r12 + 44]
    vfmadd231ps ymm10, ymm14, ymm12
    vfmadd231ps ymm11, ymm14, ymm13
    vmovaps ymm12, ymmword ptr [r10 + 128]
    vmovaps ymm13, ymmword ptr [r10 + 160]
    vbroadcastss ymm14, dword ptr [r12 + 48]
    vfmadd231ps ymm0, ymm14, ymm12
    vfmadd231ps ymm1, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r12 + 52]
    vfmadd231ps ymm2, ymm14, ymm12
    vfmadd231ps ymm3, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r12 + 56]
    vfmadd231ps ymm4, ymm14, ymm12
    vfmadd231ps ymm5, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r12 + 60]
    vfmadd231ps ymm6, ymm14, ymm12
    vfmadd231ps ymm7, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r12 + 64]
    vfmadd231ps ymm8, ymm14, ymm12
    vfmadd231ps ymm9, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r12 + 68]
    vfmadd231ps ymm10, ymm14, ymm12
    vfmadd231ps ymm11, ymm14, ymm13
    vmovaps ymm12, ymmword ptr [r10 + 192]
    vmovaps ymm13, ymmword ptr [r10 + 224]
    vbroadcastss ymm14, dword ptr [r12 + 72]
    vfmadd231ps ymm0, ymm14, ymm12
    vfmadd231ps ymm1, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r12 + 76]
    vfmadd231ps ymm2, ymm14, ymm12
    vfmadd231ps ymm3, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r12 + 80]
    vfmadd231ps ymm4, ymm14, ymm12
    vfmadd231ps ymm5, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r12 + 84]
    vfmadd231ps ymm6, ymm14, ymm12
    vfmadd231ps ymm7, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r12 + 88]
    vfmadd231ps ymm8, ymm14, ymm12
    vfmadd231ps ymm9, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r12 + 92]
    vfmadd231ps ymm10, ymm14, ymm12
    vfmadd231ps ymm11, ymm14, ymm13
    add r12, 96
    add r10, 256
    sub r13, 4
    jae 22b
23:
    # the steps past the last whole turn
    add r13, 4
    jz 25f
24:
    vmovaps ymm12, ymmword ptr [r10]
    vmovaps ymm13, ymmword ptr [r10 + 32]
    vbroadcastss ymm14, dword ptr [r12]
    vfmadd231ps ymm0, ymm14, ymm12
    vfmadd231ps ymm1, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r12 + 4]
    vfmadd231ps ymm2, ymm14, ymm12
    vfmadd231ps ymm3, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r12 + 8]
    vfmadd231ps ymm4, ymm14, ymm12
    vfmadd231ps ymm5, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r12 + 12]
    vfmadd231ps ymm6, ymm14, ymm12
    vfmadd231ps ymm7, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r12 + 16]
    vfmadd231ps ymm8, ymm14, ymm12
    vfmadd231ps ymm9, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r12 + 20]
    vfmadd231ps ymm10, ymm14, ymm12
    vfmadd231ps ymm11, ymm14, ymm13
    add r12, 24
    add r10, 64
    dec r13
    jnz 24b
25:
    mov r13, r11
    cmp r9, 6
    jb 26f
    cmp rbx, 16
    jb 26f
    vmovups ymmword ptr [r13], ymm0
    vmovups ymmword ptr [r13 + 32], ymm1
    add r13, rax
    vmovups ymmword ptr [r13], ymm2
    vmovups ymmword ptr [r13 + 32], ymm3
    add r13, rax
    vmovups ymmword ptr [r13], ymm4
    vmovups ymmword ptr [r13 + 32], ymm5
    add r13, rax
    vmovups ymmword ptr [r13], ymm6
    vmovups ymmword ptr [r13 + 32], ymm7
    add r13, rax
    vmovups ymmword ptr [r13], ymm8
    vmovups ymmword ptr [r13 + 32], ymm9
    add r13, rax
    vmovups ymmword ptr [r13], ymm10
    vmovups ymmword ptr [r13 + 32], ymm11
    add r13, rax
    jmp 27f
26:
    mov r14, rbx
    mov r15, 16
    cmp r14, r15
    cmova r14, r15
    vmovd xmm12, r14d
    vpbroadcastd ymm12, xmm12
    vpcmpgtd ymm13, ymm12, ymmword ptr [rip + {c0}]
    vpcmpgtd ymm12, ymm12, ymmword ptr [rip + {c1}]
    vmaskmovps ymmword ptr [r13], ymm13, ymm0
    vmaskmovps ymmword ptr [r13 + 32], ymm12, ymm1
    add r13, rax
    cmp r9, 1
    jbe 27f
    vmaskmovps ymmword ptr [r13], ymm13, ymm2
    vmaskmovps ymmword ptr [r13 + 32], ymm12, ymm3
    add r13, rax
    cmp r9, 2
    jbe 27f
    vmaskmovps ymmword ptr [r13], ymm13, ymm4
    vmaskmovps ymmword ptr [r13 + 32], ymm12, ymm5
    add r13, rax
    cmp r9, 3
    jbe 27f
    vmaskmovps ymmword ptr [r13], ymm13, ymm6
    vmaskmovps ymmword ptr [r13 + 32], ymm12, ymm7
    add r13, rax
    cmp r9, 4
    jbe 27f
    vmaskmovps ymmword ptr [r13], ymm13, ymm8
    vmaskmovps ymmword ptr [r13 + 32], ymm12, ymm9
    add r13, rax
    cmp r9, 5
    jbe 27f
    vmaskmovps ymmword ptr [r13], ymm13, ymm10
    vmaskmovps ymmword ptr [r13 + 32], ymm12, ymm11
    add r13, rax
27:
    add r11, 64
    sub rbx, 16
    ja 18b
    mov rdi, r12
    lea r8, [r8 + rax*4]
    lea r8, [r8 + rax*2]
    sub r9, 6
    ja 17b
    # bj reloaded
    mov rsi, qword ptr [rsp + 56]
    add rsi, 1024
    # bj spilled
    mov qword ptr [rsp + 56], rsi
    # cj reloaded
    mov rsi, qword ptr [rsp + 64]
    add rsi, 1024
    # cj spilled
    mov qword ptr [rsp + 64], rsi
    # nleft reloaded
    mov rsi, qword ptr [rsp + 72]
    sub rsi, 256
    # nleft spilled
    mov qword ptr [rsp + 72], rsi
    ja 13b
    # ak reloaded
    mov rdx, qword ptr [rsp + 24]
    add rdx, 1536
    # ak spilled
    mov qword ptr [rsp + 24], rdx
    imul rdx, rax, 384
    # bk reloaded
    mov rsi, qword ptr [rsp + 32]
    add rsi, rdx
    # bk spilled
    mov qword ptr [rsp + 32], rsi
    add rcx, 384
    # k reloaded
    mov rdx, qword ptr [rsp + 16]
    cmp rcx, rdx
    jb 2b
28:
    mov rsp, rbp
    pop rbp
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    vzeroupper
    ret
# c0, aligned to 32:
#   00 00 00 00 01 00 00 00 02 00 00 00 03 00 00 00
#   04 00 00 00 05 00 00 00 06 00 00 00 07 00 00 00
# c1, aligned to 32:
#   08 00 00 00 09 00 00 00 0a 00 00 00 0b 00 00 00
#   0c 00 00 00 0d 00 00 00 0e 00 00 00 0f 00 00 00
