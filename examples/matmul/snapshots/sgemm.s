# sgemm(gpr, gpr, gpr, gpr, gpr) -> ()
# needs: avx, avx2, fma
# frame: 147584 bytes aligned to 64, saves [rbx, r12, r13, r14, r15]
# block: 147456 bytes at rsp
# bj, spilled: 8 bytes at rsp + 147456
# nleft, spilled: 8 bytes at rsp + 147464
# a, spilled: 8 bytes at rsp + 147472
# m, spilled: 8 bytes at rsp + 147480
# cj, spilled: 8 bytes at rsp + 147488
# bk, spilled: 8 bytes at rsp + 147496
# packed, spilled: 8 bytes at rsp + 147504
# nc, spilled: 8 bytes at rsp + 147512
# k, spilled: 8 bytes at rsp + 147520
# ai, spilled: 8 bytes at rsp + 147528
# ci, spilled: 8 bytes at rsp + 147536
# c=rdi a=rsi b=rdx shape=rcx packed=r8,rcx,rsi n=rax
# ld=rcx most=rax,rsi,r10,r8,r15 pc=rax kc=rdx q=rsi,r8 col=rdi
# left=r8,r11 mask0=ymm1,ymm12,ymm13 mask1=ymm0,ymm13,ymm12 t=r9,r14,r8 lanes=ymm0,ymm12 p=r9,r10,r13
# l=r10,r13,r14 v=ymm2 mleft=rsi mc=rdi pa=r8 lda=r9
# r=rbx at=r8 s=r14,ymm14 d=r12 f=r15 pb=r9
# cs=r10 wleft=r11 ct=rbx hleft=r12 acc0=ymm0 acc1=ymm1
# acc2=ymm2 acc3=ymm3 acc4=ymm4 acc5=ymm5 acc6=ymm6 acc7=ymm7
# acc8=ymm8 acc9=ymm9 acc10=ymm10 acc11=ymm11 b0=ymm12 b1=ymm13
# bj=rdx,rax nleft=rax m=rax,rcx,rsi cj=rdx,rsi,rax bk=rdx,rdi nc=rax,rdx,r8
# k=rcx,rdx,r9,rdi ai=rsi,r10,rdi ci=rsi,r8,rdi
    push rbx
    push r12
    push r13
    push r14
    push r15
    push rbp
    mov rbp, rsp
    # the frame a page at a time, each touched, so that an overflow faults
    lea r11, [rsp - 147456]
2:
    sub rsp, 4096
    or qword ptr [rsp], 0
    cmp rsp, r11
    jne 2b
    sub rsp, 128
    and rsp, -64
    # packed spilled
    mov qword ptr [rsp + 147504], r8
    # a spilled
    mov qword ptr [rsp + 147472], rsi
    mov rax, qword ptr [rcx]
    # m spilled
    mov qword ptr [rsp + 147480], rax
    mov rax, qword ptr [rcx + 8]
    mov rcx, qword ptr [rcx + 16]
    # k spilled
    mov qword ptr [rsp + 147520], rcx
    # m reloaded
    mov rcx, qword ptr [rsp + 147480]
    test rcx, rcx
    jz 28f
    test rax, rax
    jz 28f
    # packed reloaded
    mov rcx, qword ptr [rsp + 147504]
    add rcx, 63
    # packed spilled
    mov qword ptr [rsp + 147504], rcx
    # packed reloaded
    mov rcx, qword ptr [rsp + 147504]
    and rcx, -64
    # packed spilled
    mov qword ptr [rsp + 147504], rcx
    # the bytes from a row of b or c to the next
    mov rcx, rax
    shl rcx, 2
    # panels of NC columns of b, and of c
    # bj spilled
    mov qword ptr [rsp + 147456], rdx
    mov rdx, rdi
    # cj spilled
    mov qword ptr [rsp + 147488], rdx
    # nleft spilled
    mov qword ptr [rsp + 147464], rax
3:
    # nleft reloaded
    mov rax, qword ptr [rsp + 147464]
    # nc spilled
    mov qword ptr [rsp + 147512], rax
    mov rax, 4096
    # nc reloaded
    mov rdx, qword ptr [rsp + 147512]
    cmp rdx, rax
    # nc reloaded
    mov rdx, qword ptr [rsp + 147512]
    cmova rdx, rax
    # nc spilled
    mov qword ptr [rsp + 147512], rdx
    # blocks of KC steps of k from step pc, each chain going on from the block before
    xor eax, eax
    # bj reloaded
    mov rdx, qword ptr [rsp + 147456]
    # bk spilled
    mov qword ptr [rsp + 147496], rdx
4:
    # k reloaded
    mov rdx, qword ptr [rsp + 147520]
    sub rdx, rax
    mov rsi, 256
    cmp rdx, rsi
    cmova rdx, rsi
    test rdx, rdx
    jz 7f
    # packed reloaded
    mov rsi, qword ptr [rsp + 147504]
    # bk reloaded
    mov rdi, qword ptr [rsp + 147496]
    # nc reloaded
    mov r8, qword ptr [rsp + 147512]
5:
    mov r9, r8
    mov r10, 16
    cmp r9, r10
    cmova r9, r10
    vmovd xmm0, r9d
    vpbroadcastd ymm0, xmm0
    vpcmpgtd ymm1, ymm0, ymmword ptr [rip + {c0}]
    vpcmpgtd ymm0, ymm0, ymmword ptr [rip + {c1}]
    mov r9, rdi
    mov r10, rdx
6:
    vmaskmovps ymm2, ymm1, ymmword ptr [r9]
    vmovaps ymmword ptr [rsi], ymm2
    vmaskmovps ymm2, ymm0, ymmword ptr [r9 + 32]
    vmovaps ymmword ptr [rsi + 32], ymm2
    add r9, rcx
    add rsi, 64
    dec r10
    jnz 6b
    add rdi, 64
    sub r8, 16
    ja 5b
7:
    # blocks of MC rows of a, and of c
    # a reloaded
    mov rsi, qword ptr [rsp + 147472]
    lea rsi, [rsi + rax*4]
    # ai spilled
    mov qword ptr [rsp + 147528], rsi
    # cj reloaded
    mov rsi, qword ptr [rsp + 147488]
    # ci spilled
    mov qword ptr [rsp + 147536], rsi
    # m reloaded
    mov rsi, qword ptr [rsp + 147480]
8:
    mov rdi, rsi
    mov r8, 144
    cmp rdi, r8
    cmova rdi, r8
    lea r8, [rsp]
    test rdx, rdx
    jz 16f
    # k reloaded
    mov r9, qword ptr [rsp + 147520]
    shl r9, 2
    # ai reloaded
    mov r10, qword ptr [rsp + 147528]
    mov r11, rdi
9:
    xor ebx, ebx
12:
    mov r12, r8
    mov r13, rdx
    cmp rbx, r11
    jae 14f
    mov r14, r10
13:
    mov r15d, dword ptr [r14]
    mov dword ptr [r12], r15d
    add r14, 4
    add r12, 24
    dec r13
    jnz 13b
    jmp 15f
14:
    mov dword ptr [r12], 0
    add r12, 24
    dec r13
    jnz 14b
15:
    add r10, r9
    add r8, 4
    inc rbx
    cmp rbx, 6
    jb 12b
    lea r8, [r12 - 20]
    sub r11, 6
    ja 9b
16:
    # slivers of NR columns of the panel
    # packed reloaded
    mov r8, qword ptr [rsp + 147504]
    mov r9, r8
    # ci reloaded
    mov r8, qword ptr [rsp + 147536]
    mov r10, r8
    # nc reloaded
    mov r8, qword ptr [rsp + 147512]
    mov r11, r8
17:
    lea r8, [rsp]
    mov rbx, r10
    mov r12, rdi
    # slivers of MR rows of the block, a tile of c each
18:
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
    test rax, rax
    jz 21f
    mov r13, rbx
    cmp r12, 6
    jb 19f
    cmp r11, 16
    jb 19f
    vmovups ymm0, ymmword ptr [r13]
    vmovups ymm1, ymmword ptr [r13 + 32]
    add r13, rcx
    vmovups ymm2, ymmword ptr [r13]
    vmovups ymm3, ymmword ptr [r13 + 32]
    add r13, rcx
    vmovups ymm4, ymmword ptr [r13]
    vmovups ymm5, ymmword ptr [r13 + 32]
    add r13, rcx
    vmovups ymm6, ymmword ptr [r13]
    vmovups ymm7, ymmword ptr [r13 + 32]
    add r13, rcx
    vmovups ymm8, ymmword ptr [r13]
    vmovups ymm9, ymmword ptr [r13 + 32]
    add r13, rcx
    vmovups ymm10, ymmword ptr [r13]
    vmovups ymm11, ymmword ptr [r13 + 32]
    add r13, rcx
    jmp 20f
19:
    mov r14, r11
    mov r15, 16
    cmp r14, r15
    cmova r14, r15
    vmovd xmm0, r14d
    vpbroadcastd ymm0, xmm0
    vpcmpgtd ymm12, ymm0, ymmword ptr [rip + {c0}]
    vpcmpgtd ymm13, ymm0, ymmword ptr [rip + {c1}]
    vmaskmovps ymm0, ymm12, ymmword ptr [r13]
    vmaskmovps ymm1, ymm13, ymmword ptr [r13 + 32]
    add r13, rcx
    cmp r12, 1
    jbe 20f
    vmaskmovps ymm2, ymm12, ymmword ptr [r13]
    vmaskmovps ymm3, ymm13, ymmword ptr [r13 + 32]
    add r13, rcx
    cmp r12, 2
    jbe 20f
    vmaskmovps ymm4, ymm12, ymmword ptr [r13]
    vmaskmovps ymm5, ymm13, ymmword ptr [r13 + 32]
    add r13, rcx
    cmp r12, 3
    jbe 20f
    vmaskmovps ymm6, ymm12, ymmword ptr [r13]
    vmaskmovps ymm7, ymm13, ymmword ptr [r13 + 32]
    add r13, rcx
    cmp r12, 4
    jbe 20f
    vmaskmovps ymm8, ymm12, ymmword ptr [r13]
    vmaskmovps ymm9, ymm13, ymmword ptr [r13 + 32]
    add r13, rcx
    cmp r12, 5
    jbe 20f
    vmaskmovps ymm10, ymm12, ymmword ptr [r13]
    vmaskmovps ymm11, ymm13, ymmword ptr [r13 + 32]
    add r13, rcx
20:
21:
    # UNROLL steps of k a turn
    mov r13, r9
    mov r14, rdx
    sub r14, 4
    jb 23f
22:
    vmovaps ymm12, ymmword ptr [r13]
    vmovaps ymm13, ymmword ptr [r13 + 32]
    vbroadcastss ymm14, dword ptr [r8]
    vfmadd231ps ymm0, ymm14, ymm12
    vfmadd231ps ymm1, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r8 + 4]
    vfmadd231ps ymm2, ymm14, ymm12
    vfmadd231ps ymm3, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r8 + 8]
    vfmadd231ps ymm4, ymm14, ymm12
    vfmadd231ps ymm5, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r8 + 12]
    vfmadd231ps ymm6, ymm14, ymm12
    vfmadd231ps ymm7, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r8 + 16]
    vfmadd231ps ymm8, ymm14, ymm12
    vfmadd231ps ymm9, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r8 + 20]
    vfmadd231ps ymm10, ymm14, ymm12
    vfmadd231ps ymm11, ymm14, ymm13
    vmovaps ymm12, ymmword ptr [r13 + 64]
    vmovaps ymm13, ymmword ptr [r13 + 96]
    vbroadcastss ymm14, dword ptr [r8 + 24]
    vfmadd231ps ymm0, ymm14, ymm12
    vfmadd231ps ymm1, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r8 + 28]
    vfmadd231ps ymm2, ymm14, ymm12
    vfmadd231ps ymm3, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r8 + 32]
    vfmadd231ps ymm4, ymm14, ymm12
    vfmadd231ps ymm5, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r8 + 36]
    vfmadd231ps ymm6, ymm14, ymm12
    vfmadd231ps ymm7, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r8 + 40]
    vfmadd231ps ymm8, ymm14, ymm12
    vfmadd231ps ymm9, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r8 + 44]
    vfmadd231ps ymm10, ymm14, ymm12
    vfmadd231ps ymm11, ymm14, ymm13
    vmovaps ymm12, ymmword ptr [r13 + 128]
    vmovaps ymm13, ymmword ptr [r13 + 160]
    vbroadcastss ymm14, dword ptr [r8 + 48]
    vfmadd231ps ymm0, ymm14, ymm12
    vfmadd231ps ymm1, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r8 + 52]
    vfmadd231ps ymm2, ymm14, ymm12
    vfmadd231ps ymm3, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r8 + 56]
    vfmadd231ps ymm4, ymm14, ymm12
    vfmadd231ps ymm5, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r8 + 60]
    vfmadd231ps ymm6, ymm14, ymm12
    vfmadd231ps ymm7, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r8 + 64]
    vfmadd231ps ymm8, ymm14, ymm12
    vfmadd231ps ymm9, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r8 + 68]
    vfmadd231ps ymm10, ymm14, ymm12
    vfmadd231ps ymm11, ymm14, ymm13
    vmovaps ymm12, ymmword ptr [r13 + 192]
    vmovaps ymm13, ymmword ptr [r13 + 224]
    vbroadcastss ymm14, dword ptr [r8 + 72]
    vfmadd231ps ymm0, ymm14, ymm12
    vfmadd231ps ymm1, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r8 + 76]
    vfmadd231ps ymm2, ymm14, ymm12
    vfmadd231ps ymm3, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r8 + 80]
    vfmadd231ps ymm4, ymm14, ymm12
    vfmadd231ps ymm5, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r8 + 84]
    vfmadd231ps ymm6, ymm14, ymm12
    vfmadd231ps ymm7, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r8 + 88]
    vfmadd231ps ymm8, ymm14, ymm12
    vfmadd231ps ymm9, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r8 + 92]
    vfmadd231ps ymm10, ymm14, ymm12
    vfmadd231ps ymm11, ymm14, ymm13
    add r8, 96
    add r13, 256
    sub r14, 4
    jae 22b
23:
    # the steps past the last whole turn
    add r14, 4
    jz 25f
24:
    vmovaps ymm12, ymmword ptr [r13]
    vmovaps ymm13, ymmword ptr [r13 + 32]
    vbroadcastss ymm14, dword ptr [r8]
    vfmadd231ps ymm0, ymm14, ymm12
    vfmadd231ps ymm1, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r8 + 4]
    vfmadd231ps ymm2, ymm14, ymm12
    vfmadd231ps ymm3, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r8 + 8]
    vfmadd231ps ymm4, ymm14, ymm12
    vfmadd231ps ymm5, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r8 + 12]
    vfmadd231ps ymm6, ymm14, ymm12
    vfmadd231ps ymm7, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r8 + 16]
    vfmadd231ps ymm8, ymm14, ymm12
    vfmadd231ps ymm9, ymm14, ymm13
    vbroadcastss ymm14, dword ptr [r8 + 20]
    vfmadd231ps ymm10, ymm14, ymm12
    vfmadd231ps ymm11, ymm14, ymm13
    add r8, 24
    add r13, 64
    dec r14
    jnz 24b
25:
    mov r13, rbx
    cmp r12, 6
    jb 26f
    cmp r11, 16
    jb 26f
    vmovups ymmword ptr [r13], ymm0
    vmovups ymmword ptr [r13 + 32], ymm1
    add r13, rcx
    vmovups ymmword ptr [r13], ymm2
    vmovups ymmword ptr [r13 + 32], ymm3
    add r13, rcx
    vmovups ymmword ptr [r13], ymm4
    vmovups ymmword ptr [r13 + 32], ymm5
    add r13, rcx
    vmovups ymmword ptr [r13], ymm6
    vmovups ymmword ptr [r13 + 32], ymm7
    add r13, rcx
    vmovups ymmword ptr [r13], ymm8
    vmovups ymmword ptr [r13 + 32], ymm9
    add r13, rcx
    vmovups ymmword ptr [r13], ymm10
    vmovups ymmword ptr [r13 + 32], ymm11
    add r13, rcx
    jmp 27f
26:
    mov r14, r11
    mov r15, 16
    cmp r14, r15
    cmova r14, r15
    vmovd xmm12, r14d
    vpbroadcastd ymm12, xmm12
    vpcmpgtd ymm13, ymm12, ymmword ptr [rip + {c0}]
    vpcmpgtd ymm12, ymm12, ymmword ptr [rip + {c1}]
    vmaskmovps ymmword ptr [r13], ymm13, ymm0
    vmaskmovps ymmword ptr [r13 + 32], ymm12, ymm1
    add r13, rcx
    cmp r12, 1
    jbe 27f
    vmaskmovps ymmword ptr [r13], ymm13, ymm2
    vmaskmovps ymmword ptr [r13 + 32], ymm12, ymm3
    add r13, rcx
    cmp r12, 2
    jbe 27f
    vmaskmovps ymmword ptr [r13], ymm13, ymm4
    vmaskmovps ymmword ptr [r13 + 32], ymm12, ymm5
    add r13, rcx
    cmp r12, 3
    jbe 27f
    vmaskmovps ymmword ptr [r13], ymm13, ymm6
    vmaskmovps ymmword ptr [r13 + 32], ymm12, ymm7
    add r13, rcx
    cmp r12, 4
    jbe 27f
    vmaskmovps ymmword ptr [r13], ymm13, ymm8
    vmaskmovps ymmword ptr [r13 + 32], ymm12, ymm9
    add r13, rcx
    cmp r12, 5
    jbe 27f
    vmaskmovps ymmword ptr [r13], ymm13, ymm10
    vmaskmovps ymmword ptr [r13 + 32], ymm12, ymm11
    add r13, rcx
27:
    lea rbx, [rbx + rcx*4]
    lea rbx, [rbx + rcx*2]
    sub r12, 6
    ja 18b
    mov r8, rdx
    shl r8, 6
    add r9, r8
    add r10, 64
    sub r11, 16
    ja 17b
    # k reloaded
    mov rdi, qword ptr [rsp + 147520]
    imul r8, rdi, 576
    # ai reloaded
    mov rdi, qword ptr [rsp + 147528]
    add rdi, r8
    # ai spilled
    mov qword ptr [rsp + 147528], rdi
    imul r8, rcx, 144
    # ci reloaded
    mov rdi, qword ptr [rsp + 147536]
    add rdi, r8
    # ci spilled
    mov qword ptr [rsp + 147536], rdi
    sub rsi, 144
    ja 8b
    imul r8, rcx, 256
    # bk reloaded
    mov rdx, qword ptr [rsp + 147496]
    add rdx, r8
    # bk spilled
    mov qword ptr [rsp + 147496], rdx
    add rax, 256
    # k reloaded
    mov rdx, qword ptr [rsp + 147520]
    cmp rax, rdx
    jb 4b
    # bj reloaded
    mov rax, qword ptr [rsp + 147456]
    add rax, 16384
    # bj spilled
    mov qword ptr [rsp + 147456], rax
    # cj reloaded
    mov rax, qword ptr [rsp + 147488]
    add rax, 16384
    # cj spilled
    mov qword ptr [rsp + 147488], rax
    # nleft reloaded
    mov rax, qword ptr [rsp + 147464]
    sub rax, 4096
    # nleft spilled
    mov qword ptr [rsp + 147464], rax
    ja 3b
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
