# sgemm(gpr, gpr, gpr, gpr, gpr, gpr) -> ()
# needs: avx, avx2, fma
# frame: 196672 bytes aligned to 64, saves [rbx, r12, r13, r14, r15]
# block: 196608 bytes at rsp
# m, spilled: 8 bytes at rsp + 196608
# c, spilled: 8 bytes at rsp + 196616
# ak, spilled: 8 bytes at rsp + 196624
# bk, spilled: 8 bytes at rsp + 196632
# packed, spilled: 8 bytes at rsp + 196640
# k, spilled: 8 bytes at rsp + 196648
# shape, spilled: 8 bytes at rsp + 196656
# counters, spilled: 8 bytes at rsp + 196664
# c=rdi,r11 a=rsi b=rdx shape=rcx,rax,rdx,rdi,rsi,r8,r9,r10 packed=r8,r9,r11 n=rax,rbx
# ld=rax pc=rcx kc=rdx most=rsi,r8,r13,r15 part=rsi sliver=r8
# row=rsi mine=rdi lda=r9 q=r8,r11 p=rsi,r12,r13 left=rdi,rbx
# s=r10,ymm14 s3=r11 l=rbx,r13 f=r12 r=r10 at=r8
# from_row=r12 d=r11 g=r13 total=rsi piece=rsi cj=rdi
# nc=r8 ri=r9 hleft=rsi bj=r10 to=r11 col=r10
# mask0=ymm1,ymm12,ymm13 mask1=ymm0,ymm13,ymm12 t=r12,r14,rdx lanes=ymm0,ymm12 v=ymm2 pa=r10
# ci=rdi pb=r9 ct=r11 wleft=rbx x=r12 acc0=ymm0
# acc1=ymm1 acc2=ymm2 acc3=ymm3 acc4=ymm4 acc5=ymm5 acc6=ymm6
# acc7=ymm7 acc8=ymm8 acc9=ymm9 acc10=ymm10 acc11=ymm11 b0=ymm12
# b1=ymm13 grid=r9 m=rax,rdi ak=rsi,r9,rdx bk=rdx,r10,rsi k=rcx,rdx,r9
# counters=rdx,rdi,rsi
    push rbx
    push r12
    push r13
    push r14
    push r15
    push rbp
    mov rbp, rsp
    # the frame a page at a time, each touched, so that an overflow faults
    lea r11, [rsp - 196608]
2:
    sub rsp, 4096
    or qword ptr [rsp], 0
    cmp rsp, r11
    jne 2b
    sub rsp, 64
    and rsp, -64
    # shape spilled
    mov qword ptr [rsp + 196656], rcx
    # packed spilled
    mov qword ptr [rsp + 196640], r8
    # c spilled
    mov qword ptr [rsp + 196616], rdi
    # shape reloaded
    mov rax, qword ptr [rsp + 196656]
    mov rax, qword ptr [rax]
    # m spilled
    mov qword ptr [rsp + 196608], rax
    # shape reloaded
    mov rax, qword ptr [rsp + 196656]
    mov rax, qword ptr [rax + 8]
    # shape reloaded
    mov rcx, qword ptr [rsp + 196656]
    mov rcx, qword ptr [rcx + 16]
    # k spilled
    mov qword ptr [rsp + 196648], rcx
    # the bytes from a row of b or c to the next
    shl rax, 2
    # blocks of KC steps of k from step pc, each chain going on from the block before
    xor ecx, ecx
    # ak spilled
    mov qword ptr [rsp + 196624], rsi
    # bk spilled
    mov qword ptr [rsp + 196632], rdx
    # a block's counters, a line each: parts taken, +64 packed; +128 pieces taken, +192 done
    # shape reloaded
    mov rdx, qword ptr [rsp + 196656]
    mov rdx, qword ptr [rdx + 40]
    # counters spilled
    mov qword ptr [rsp + 196664], rdx
3:
    # k reloaded
    mov rdx, qword ptr [rsp + 196648]
    sub rdx, rcx
    mov rsi, 768
    cmp rdx, rsi
    cmova rdx, rsi
    # parts of PACK rows of a, packed by the first thread to come for each
4:
    mov rsi, 1
    # counters reloaded
    mov rdi, qword ptr [rsp + 196664]
    lock xadd qword ptr [rdi], rsi
    # shape reloaded
    mov rdi, qword ptr [rsp + 196656]
    cmp rsi, qword ptr [rdi + 24]
    jae 15f
    imul rsi, rsi, 24
    # m reloaded
    mov rdi, qword ptr [rsp + 196608]
    sub rdi, rsi
    mov r8, 24
    cmp rdi, r8
    cmova rdi, r8
    mov r8, rsi
    imul r8, rdx
    # packed reloaded
    mov r9, qword ptr [rsp + 196640]
    lea r8, [r9 + r8*4]
    # k reloaded
    mov r9, qword ptr [rsp + 196648]
    imul rsi, r9
    # ak reloaded
    mov r9, qword ptr [rsp + 196624]
    lea rsi, [r9 + rsi*4]
    test rdx, rdx
    jz 14f
    test rdi, rdi
    jz 14f
    # k reloaded
    mov r9, qword ptr [rsp + 196648]
    shl r9, 2
5:
    cmp rdi, 6
    jb 7f
    # a whole sliver, a step at a time: rows 0 to 2 from s, and 3 to 5 from s3
    mov r10, rsi
    lea r11, [rsi + r9*2]
    add r11, r9
    mov rbx, rdx
6:
    mov r12d, dword ptr [r10]
    mov dword ptr [r8], r12d
    mov r12d, dword ptr [r10 + r9*1]
    mov dword ptr [r8 + 4], r12d
    mov r12d, dword ptr [r10 + r9*2]
    mov dword ptr [r8 + 8], r12d
    mov r12d, dword ptr [r11]
    mov dword ptr [r8 + 12], r12d
    mov r12d, dword ptr [r11 + r9*1]
    mov dword ptr [r8 + 16], r12d
    mov r12d, dword ptr [r11 + r9*2]
    mov dword ptr [r8 + 20], r12d
    add r10, 4
    add r11, 4
    add r8, 24
    dec rbx
    jnz 6b
    lea rsi, [rsi + r9*4]
    lea rsi, [rsi + r9*2]
    sub rdi, 6
    ja 5b
    jmp 14f
    # the last sliver, short of MR rows: a row at a time, then zeros
7:
    xor r10d, r10d
8:
    mov r11, r8
    mov rbx, rdx
    cmp r10, rdi
    jae 12f
    mov r12, rsi
9:
    mov r13d, dword ptr [r12]
    mov dword ptr [r11], r13d
    add r12, 4
    add r11, 24
    dec rbx
    jnz 9b
    jmp 13f
12:
    mov dword ptr [r11], 0
    add r11, 24
    dec rbx
    jnz 12b
13:
    add rsi, r9
    add r8, 4
    inc r10
    cmp r10, 6
    jb 8b
14:
    # counters reloaded
    mov rsi, qword ptr [rsp + 196664]
    lock inc qword ptr [rsi + 64]
    jmp 4b
15:
    # shape reloaded
    mov rsi, qword ptr [rsp + 196656]
    mov rsi, qword ptr [rsi + 24]
16:
    # counters reloaded
    mov rdi, qword ptr [rsp + 196664]
    cmp qword ptr [rdi + 64], rsi
    jae 17f
    pause
    jmp 16b
17:
    # pieces of c, columns and rows, each taken by the first thread to come for it
18:
    mov rsi, 1
    # counters reloaded
    mov rdi, qword ptr [rsp + 196664]
    lock xadd qword ptr [rdi + 128], rsi
    # shape reloaded
    mov rdi, qword ptr [rsp + 196656]
    cmp rsi, qword ptr [rdi + 32]
    jae 33f
    shl rsi, 5
    # shape reloaded
    mov rdi, qword ptr [rsp + 196656]
    mov rdi, qword ptr [rdi + rsi*1 + 48]
    # shape reloaded
    mov r8, qword ptr [rsp + 196656]
    mov r8, qword ptr [r8 + rsi*1 + 56]
    # shape reloaded
    mov r9, qword ptr [rsp + 196656]
    mov r9, qword ptr [r9 + rsi*1 + 64]
    # shape reloaded
    mov r10, qword ptr [rsp + 196656]
    mov rsi, qword ptr [r10 + rsi*1 + 72]
    # bk reloaded
    mov r10, qword ptr [rsp + 196632]
    lea r10, [r10 + rdi*4]
    # c reloaded
    mov r11, qword ptr [rsp + 196616]
    lea rdi, [r11 + rdi*4]
    lea r11, [rsp]
    test rdx, rdx
    jz 21f
    mov rbx, r8
19:
    mov r12, rbx
    mov r13, 16
    cmp r12, r13
    cmova r12, r13
    vmovd xmm0, r12d
    vpbroadcastd ymm0, xmm0
    vpcmpgtd ymm1, ymm0, ymmword ptr [rip + {c0}]
    vpcmpgtd ymm0, ymm0, ymmword ptr [rip + {c1}]
    mov r12, r10
    mov r13, rdx
20:
    vmaskmovps ymm2, ymm1, ymmword ptr [r12]
    vmovaps ymmword ptr [r11], ymm2
    vmaskmovps ymm2, ymm0, ymmword ptr [r12 + 32]
    vmovaps ymmword ptr [r11 + 32], ymm2
    add r12, rax
    add r11, 64
    dec r13
    jnz 20b
    add r10, 64
    sub rbx, 16
    ja 19b
21:
    # slivers of MR rows of a, and of c
    mov r10, r9
    imul r10, rdx
    # packed reloaded
    mov r11, qword ptr [rsp + 196640]
    lea r10, [r11 + r10*4]
    imul r9, rax
    lea rdi, [rdi + r9*1]
22:
    # slivers of NR columns of the piece, a tile of c each
    lea r9, [rsp]
    mov r11, rdi
    mov rbx, r8
23:
    mov r12, r10
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
    jz 26f
    mov r13, r11
    cmp rsi, 6
    jb 24f
    cmp rbx, 16
    jb 24f
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
    jmp 25f
24:
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
    cmp rsi, 1
    jbe 25f
    vmaskmovps ymm2, ymm12, ymmword ptr [r13]
    vmaskmovps ymm3, ymm13, ymmword ptr [r13 + 32]
    add r13, rax
    cmp rsi, 2
    jbe 25f
    vmaskmovps ymm4, ymm12, ymmword ptr [r13]
    vmaskmovps ymm5, ymm13, ymmword ptr [r13 + 32]
    add r13, rax
    cmp rsi, 3
    jbe 25f
    vmaskmovps ymm6, ymm12, ymmword ptr [r13]
    vmaskmovps ymm7, ymm13, ymmword ptr [r13 + 32]
    add r13, rax
    cmp rsi, 4
    jbe 25f
    vmaskmovps ymm8, ymm12, ymmword ptr [r13]
    vmaskmovps ymm9, ymm13, ymmword ptr [r13 + 32]
    add r13, rax
    cmp rsi, 5
    jbe 25f
    vmaskmovps ymm10, ymm12, ymmword ptr [r13]
    vmaskmovps ymm11, ymm13, ymmword ptr [r13 + 32]
    add r13, rax
25:
26:
    # UNROLL steps of k a turn
    mov r13, rdx
    sub r13, 4
    jb 28f
27:
    vmovaps ymm12, ymmword ptr [r9]
    vmovaps ymm13, ymmword ptr [r9 + 32]
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
    vmovaps ymm12, ymmword ptr [r9 + 64]
    vmovaps ymm13, ymmword ptr [r9 + 96]
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
    vmovaps ymm12, ymmword ptr [r9 + 128]
    vmovaps ymm13, ymmword ptr [r9 + 160]
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
    vmovaps ymm12, ymmword ptr [r9 + 192]
    vmovaps ymm13, ymmword ptr [r9 + 224]
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
    add r9, 256
    sub r13, 4
    jae 27b
28:
    # the steps past the last whole turn
    add r13, 4
    jz 30f
29:
    vmovaps ymm12, ymmword ptr [r9]
    vmovaps ymm13, ymmword ptr [r9 + 32]
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
    add r9, 64
    dec r13
    jnz 29b
30:
    mov r13, r11
    cmp rsi, 6
    jb 31f
    cmp rbx, 16
    jb 31f
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
    jmp 32f
31:
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
    cmp rsi, 1
    jbe 32f
    vmaskmovps ymmword ptr [r13], ymm13, ymm2
    vmaskmovps ymmword ptr [r13 + 32], ymm12, ymm3
    add r13, rax
    cmp rsi, 2
    jbe 32f
    vmaskmovps ymmword ptr [r13], ymm13, ymm4
    vmaskmovps ymmword ptr [r13 + 32], ymm12, ymm5
    add r13, rax
    cmp rsi, 3
    jbe 32f
    vmaskmovps ymmword ptr [r13], ymm13, ymm6
    vmaskmovps ymmword ptr [r13 + 32], ymm12, ymm7
    add r13, rax
    cmp rsi, 4
    jbe 32f
    vmaskmovps ymmword ptr [r13], ymm13, ymm8
    vmaskmovps ymmword ptr [r13 + 32], ymm12, ymm9
    add r13, rax
    cmp rsi, 5
    jbe 32f
    vmaskmovps ymmword ptr [r13], ymm13, ymm10
    vmaskmovps ymmword ptr [r13 + 32], ymm12, ymm11
    add r13, rax
32:
    add r11, 64
    sub rbx, 16
    ja 23b
    mov r10, r12
    lea rdi, [rdi + rax*4]
    lea rdi, [rdi + rax*2]
    sub rsi, 6
    ja 22b
    # counters reloaded
    mov rsi, qword ptr [rsp + 196664]
    lock inc qword ptr [rsi + 192]
    jmp 18b
33:
    # shape reloaded
    mov rdx, qword ptr [rsp + 196656]
    mov rsi, qword ptr [rdx + 32]
34:
    # counters reloaded
    mov rdx, qword ptr [rsp + 196664]
    cmp qword ptr [rdx + 192], rsi
    jae 35f
    pause
    jmp 34b
35:
    # ak reloaded
    mov rdx, qword ptr [rsp + 196624]
    add rdx, 3072
    # ak spilled
    mov qword ptr [rsp + 196624], rdx
    imul rdx, rax, 768
    # bk reloaded
    mov rsi, qword ptr [rsp + 196632]
    add rsi, rdx
    # bk spilled
    mov qword ptr [rsp + 196632], rsi
    # counters reloaded
    mov rdx, qword ptr [rsp + 196664]
    add rdx, 256
    # counters spilled
    mov qword ptr [rsp + 196664], rdx
    add rcx, 768
    # k reloaded
    mov rdx, qword ptr [rsp + 196648]
    cmp rcx, rdx
    jb 3b
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
