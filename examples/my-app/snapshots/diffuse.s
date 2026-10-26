# diffuse(gpr, gpr, gpr, gpr, gpr) -> ()
# frame: 0 bytes aligned to 0, saves [rbx]
# x=rdi y=rsi n=rdx steps=rcx from=rax to=rsi
# c=r9 i=r10 end=r11 j=rbx quarter=ymm0 grid=r8
# v=ymm1 l=ymm2 r=ymm3 barrier=r9 generation=r10 arrived=r11
    push rbx
    mov rax, rdi
    vbroadcastss ymm0, dword ptr [rip + {c0}]
    test rcx, rcx
    jz 16f
2:
    mov r9, qword ptr [r8]
3:
    mov r10, r9
    shl r10, 10
    cmp r10, rdx
    jae 9f
    lea r11, [r10 + 1024]
    cmp r11, rdx
    jbe 4f
    mov r11, rdx
4:
    cmp r10, r11
    jz 8f
    # eight floats at once, if their neighbours are all there
    test r10, r10
    jz 5f
    lea rbx, [r10 + 9]
    cmp rbx, rdx
    ja 5f
    lea rbx, [r10 + 8]
    cmp rbx, r11
    ja 5f
    vmovups ymm1, ymmword ptr [rax + r10*4]
    vmovups ymm2, ymmword ptr [rax + r10*4 - 4]
    vmovups ymm3, ymmword ptr [rax + r10*4 + 4]
    vsubps ymm2, ymm2, ymm1
    vmulps ymm2, ymm2, ymm0
    vsubps ymm3, ymm3, ymm1
    vmulps ymm3, ymm3, ymm0
    vaddps ymm1, ymm1, ymm2
    vaddps ymm1, ymm1, ymm3
    vmovups ymmword ptr [rsi + r10*4], ymm1
    add r10, 8
    jmp 4b
5:
    # one float, an end's missing neighbour the end itself
    vmovss xmm1, dword ptr [rax + r10*4]
    vmovaps xmm2, xmm1
    vmovaps xmm3, xmm1
    test r10, r10
    jz 6f
    vmovss xmm2, dword ptr [rax + r10*4 - 4]
6:
    lea rbx, [r10 + 1]
    cmp rbx, rdx
    jz 7f
    vmovss xmm3, dword ptr [rax + r10*4 + 4]
7:
    vsubss xmm2, xmm2, xmm1
    vmulss xmm2, xmm2, xmm0
    vsubss xmm3, xmm3, xmm1
    vmulss xmm3, xmm3, xmm0
    vaddss xmm1, xmm1, xmm2
    vaddss xmm1, xmm1, xmm3
    vmovss dword ptr [rsi + r10*4], xmm1
    inc r10
    jmp 4b
8:
    add r9, qword ptr [r8 + 8]
    jmp 3b
9:
    # grid.sync()
    mov r9, qword ptr [r8 + 16]
    mov r10, qword ptr [r9 + 8]
    mov r11, 1
    lock xadd qword ptr [r9], r11
    inc r11
    cmp r11, qword ptr [r8 + 8]
    jnz 12f
    # the last to come lets the others go
    mov qword ptr [r9], 0
    inc r10
    mov qword ptr [r9 + 8], r10
    jmp 13f
12:
    # the others wait for it
    pause
    cmp r10, qword ptr [r9 + 8]
    jz 12b
13:
    mov rbx, rax
    mov rax, rsi
    mov rsi, rbx
    dec rcx
    jnz 2b
    # after an odd number of steps, the thread's chunks back to `x`
    cmp rax, rdi
    jz 16f
    mov r9, qword ptr [r8]
14:
    mov r10, r9
    shl r10, 10
    cmp r10, rdx
    jae 16f
    lea r11, [r10 + 1024]
    cmp r11, rdx
    jbe 15f
    mov r11, rdx
15:
    vmovss xmm1, dword ptr [rax + r10*4]
    vmovss dword ptr [rdi + r10*4], xmm1
    inc r10
    cmp r10, r11
    jnz 15b
    add r9, qword ptr [r8 + 8]
    jmp 14b
16:
    pop rbx
    vzeroupper
    ret
# c0, aligned to 4:
#   00 00 80 3e
