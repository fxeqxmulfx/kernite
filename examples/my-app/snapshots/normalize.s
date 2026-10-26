# normalize(gpr, gpr, gpr, gpr) -> ()
# frame: 4096 bytes aligned to 32, saves []
# tail: 4096 bytes at rsp
# x=rdi sums=rsi n=rdx whole=rax rest=r8 c=r9
# p=r10 i=r11 grid=rcx f=ymm0 tree0=ymm0 tree1=ymm1
# tree2=ymm2 tree3=ymm3 tree4=ymm4 tree5=ymm5 tree6=ymm6 tree7=ymm7
# v0=ymm0,ymm1,ymm2,ymm3,ymm4,ymm5,ymm6,ymm7 v1=ymm1,ymm2,ymm3,ymm4,ymm5,ymm6,ymm7,ymm8 v2=ymm2,ymm3,ymm4,ymm5,ymm6,ymm7,ymm8,ymm9 v3=ymm3,ymm4,ymm5,ymm6,ymm7,ymm8,ymm9,ymm10 s=ymm1 high=ymm1
# pair=ymm0 barrier=r9 generation=r10 arrived=r11 all=rdx d=ymm0
# v=ymm2
    push rbp
    mov rbp, rsp
    # the frame a page at a time, each touched, so that an overflow faults
    sub rsp, 4096
    or qword ptr [rsp], 0
    and rsp, -32
    mov rax, rdx
    shr rax, 10
    mov r8, rdx
    and r8, 1023
    # each of the thread's chunks summed
    mov r9, qword ptr [rcx]
2:
    mov r10, r9
    shl r10, 12
    add r10, rdi
    cmp r9, rax
    jb 5f
    jnz 6f
    test r8, r8
    jz 6f
    # the last chunk, partial, copied to the stack, zeros after it
    vxorps xmm0, xmm0, xmm0
    xor r11d, r11d
3:
    vmovaps ymmword ptr [rsp + r11*1], ymm0
    vmovaps ymmword ptr [rsp + r11*1 + 32], ymm0
    vmovaps ymmword ptr [rsp + r11*1 + 64], ymm0
    vmovaps ymmword ptr [rsp + r11*1 + 96], ymm0
    add r11, 128
    cmp r11, 4096
    jnz 3b
    xor r11d, r11d
4:
    vmovss xmm0, dword ptr [r10 + r11*4]
    vmovss dword ptr [rsp + r11*4], xmm0
    inc r11
    cmp r11, r8
    jnz 4b
    lea r10, [rsp]
5:
    # a GPU thread's four floats to a lane, then the tree's first two halvings
    vmovups ymm0, ymmword ptr [r10]
    vaddps ymm0, ymm0, ymmword ptr [r10 + 1024]
    vaddps ymm0, ymm0, ymmword ptr [r10 + 2048]
    vaddps ymm0, ymm0, ymmword ptr [r10 + 3072]
    vmovups ymm1, ymmword ptr [r10 + 256]
    vaddps ymm1, ymm1, ymmword ptr [r10 + 1280]
    vaddps ymm1, ymm1, ymmword ptr [r10 + 2304]
    vaddps ymm1, ymm1, ymmword ptr [r10 + 3328]
    vmovups ymm2, ymmword ptr [r10 + 512]
    vaddps ymm2, ymm2, ymmword ptr [r10 + 1536]
    vaddps ymm2, ymm2, ymmword ptr [r10 + 2560]
    vaddps ymm2, ymm2, ymmword ptr [r10 + 3584]
    vmovups ymm3, ymmword ptr [r10 + 768]
    vaddps ymm3, ymm3, ymmword ptr [r10 + 1792]
    vaddps ymm3, ymm3, ymmword ptr [r10 + 2816]
    vaddps ymm3, ymm3, ymmword ptr [r10 + 3840]
    vaddps ymm0, ymm0, ymm2
    vaddps ymm1, ymm1, ymm3
    vaddps ymm0, ymm0, ymm1
    vmovups ymm1, ymmword ptr [r10 + 32]
    vaddps ymm1, ymm1, ymmword ptr [r10 + 1056]
    vaddps ymm1, ymm1, ymmword ptr [r10 + 2080]
    vaddps ymm1, ymm1, ymmword ptr [r10 + 3104]
    vmovups ymm2, ymmword ptr [r10 + 288]
    vaddps ymm2, ymm2, ymmword ptr [r10 + 1312]
    vaddps ymm2, ymm2, ymmword ptr [r10 + 2336]
    vaddps ymm2, ymm2, ymmword ptr [r10 + 3360]
    vmovups ymm3, ymmword ptr [r10 + 544]
    vaddps ymm3, ymm3, ymmword ptr [r10 + 1568]
    vaddps ymm3, ymm3, ymmword ptr [r10 + 2592]
    vaddps ymm3, ymm3, ymmword ptr [r10 + 3616]
    vmovups ymm4, ymmword ptr [r10 + 800]
    vaddps ymm4, ymm4, ymmword ptr [r10 + 1824]
    vaddps ymm4, ymm4, ymmword ptr [r10 + 2848]
    vaddps ymm4, ymm4, ymmword ptr [r10 + 3872]
    vaddps ymm1, ymm1, ymm3
    vaddps ymm2, ymm2, ymm4
    vaddps ymm1, ymm1, ymm2
    vmovups ymm2, ymmword ptr [r10 + 64]
    vaddps ymm2, ymm2, ymmword ptr [r10 + 1088]
    vaddps ymm2, ymm2, ymmword ptr [r10 + 2112]
    vaddps ymm2, ymm2, ymmword ptr [r10 + 3136]
    vmovups ymm3, ymmword ptr [r10 + 320]
    vaddps ymm3, ymm3, ymmword ptr [r10 + 1344]
    vaddps ymm3, ymm3, ymmword ptr [r10 + 2368]
    vaddps ymm3, ymm3, ymmword ptr [r10 + 3392]
    vmovups ymm4, ymmword ptr [r10 + 576]
    vaddps ymm4, ymm4, ymmword ptr [r10 + 1600]
    vaddps ymm4, ymm4, ymmword ptr [r10 + 2624]
    vaddps ymm4, ymm4, ymmword ptr [r10 + 3648]
    vmovups ymm5, ymmword ptr [r10 + 832]
    vaddps ymm5, ymm5, ymmword ptr [r10 + 1856]
    vaddps ymm5, ymm5, ymmword ptr [r10 + 2880]
    vaddps ymm5, ymm5, ymmword ptr [r10 + 3904]
    vaddps ymm2, ymm2, ymm4
    vaddps ymm3, ymm3, ymm5
    vaddps ymm2, ymm2, ymm3
    vmovups ymm3, ymmword ptr [r10 + 96]
    vaddps ymm3, ymm3, ymmword ptr [r10 + 1120]
    vaddps ymm3, ymm3, ymmword ptr [r10 + 2144]
    vaddps ymm3, ymm3, ymmword ptr [r10 + 3168]
    vmovups ymm4, ymmword ptr [r10 + 352]
    vaddps ymm4, ymm4, ymmword ptr [r10 + 1376]
    vaddps ymm4, ymm4, ymmword ptr [r10 + 2400]
    vaddps ymm4, ymm4, ymmword ptr [r10 + 3424]
    vmovups ymm5, ymmword ptr [r10 + 608]
    vaddps ymm5, ymm5, ymmword ptr [r10 + 1632]
    vaddps ymm5, ymm5, ymmword ptr [r10 + 2656]
    vaddps ymm5, ymm5, ymmword ptr [r10 + 3680]
    vmovups ymm6, ymmword ptr [r10 + 864]
    vaddps ymm6, ymm6, ymmword ptr [r10 + 1888]
    vaddps ymm6, ymm6, ymmword ptr [r10 + 2912]
    vaddps ymm6, ymm6, ymmword ptr [r10 + 3936]
    vaddps ymm3, ymm3, ymm5
    vaddps ymm4, ymm4, ymm6
    vaddps ymm3, ymm3, ymm4
    vmovups ymm4, ymmword ptr [r10 + 128]
    vaddps ymm4, ymm4, ymmword ptr [r10 + 1152]
    vaddps ymm4, ymm4, ymmword ptr [r10 + 2176]
    vaddps ymm4, ymm4, ymmword ptr [r10 + 3200]
    vmovups ymm5, ymmword ptr [r10 + 384]
    vaddps ymm5, ymm5, ymmword ptr [r10 + 1408]
    vaddps ymm5, ymm5, ymmword ptr [r10 + 2432]
    vaddps ymm5, ymm5, ymmword ptr [r10 + 3456]
    vmovups ymm6, ymmword ptr [r10 + 640]
    vaddps ymm6, ymm6, ymmword ptr [r10 + 1664]
    vaddps ymm6, ymm6, ymmword ptr [r10 + 2688]
    vaddps ymm6, ymm6, ymmword ptr [r10 + 3712]
    vmovups ymm7, ymmword ptr [r10 + 896]
    vaddps ymm7, ymm7, ymmword ptr [r10 + 1920]
    vaddps ymm7, ymm7, ymmword ptr [r10 + 2944]
    vaddps ymm7, ymm7, ymmword ptr [r10 + 3968]
    vaddps ymm4, ymm4, ymm6
    vaddps ymm5, ymm5, ymm7
    vaddps ymm4, ymm4, ymm5
    vmovups ymm5, ymmword ptr [r10 + 160]
    vaddps ymm5, ymm5, ymmword ptr [r10 + 1184]
    vaddps ymm5, ymm5, ymmword ptr [r10 + 2208]
    vaddps ymm5, ymm5, ymmword ptr [r10 + 3232]
    vmovups ymm6, ymmword ptr [r10 + 416]
    vaddps ymm6, ymm6, ymmword ptr [r10 + 1440]
    vaddps ymm6, ymm6, ymmword ptr [r10 + 2464]
    vaddps ymm6, ymm6, ymmword ptr [r10 + 3488]
    vmovups ymm7, ymmword ptr [r10 + 672]
    vaddps ymm7, ymm7, ymmword ptr [r10 + 1696]
    vaddps ymm7, ymm7, ymmword ptr [r10 + 2720]
    vaddps ymm7, ymm7, ymmword ptr [r10 + 3744]
    vmovups ymm8, ymmword ptr [r10 + 928]
    vaddps ymm8, ymm8, ymmword ptr [r10 + 1952]
    vaddps ymm8, ymm8, ymmword ptr [r10 + 2976]
    vaddps ymm8, ymm8, ymmword ptr [r10 + 4000]
    vaddps ymm5, ymm5, ymm7
    vaddps ymm6, ymm6, ymm8
    vaddps ymm5, ymm5, ymm6
    vmovups ymm6, ymmword ptr [r10 + 192]
    vaddps ymm6, ymm6, ymmword ptr [r10 + 1216]
    vaddps ymm6, ymm6, ymmword ptr [r10 + 2240]
    vaddps ymm6, ymm6, ymmword ptr [r10 + 3264]
    vmovups ymm7, ymmword ptr [r10 + 448]
    vaddps ymm7, ymm7, ymmword ptr [r10 + 1472]
    vaddps ymm7, ymm7, ymmword ptr [r10 + 2496]
    vaddps ymm7, ymm7, ymmword ptr [r10 + 3520]
    vmovups ymm8, ymmword ptr [r10 + 704]
    vaddps ymm8, ymm8, ymmword ptr [r10 + 1728]
    vaddps ymm8, ymm8, ymmword ptr [r10 + 2752]
    vaddps ymm8, ymm8, ymmword ptr [r10 + 3776]
    vmovups ymm9, ymmword ptr [r10 + 960]
    vaddps ymm9, ymm9, ymmword ptr [r10 + 1984]
    vaddps ymm9, ymm9, ymmword ptr [r10 + 3008]
    vaddps ymm9, ymm9, ymmword ptr [r10 + 4032]
    vaddps ymm6, ymm6, ymm8
    vaddps ymm7, ymm7, ymm9
    vaddps ymm6, ymm6, ymm7
    vmovups ymm7, ymmword ptr [r10 + 224]
    vaddps ymm7, ymm7, ymmword ptr [r10 + 1248]
    vaddps ymm7, ymm7, ymmword ptr [r10 + 2272]
    vaddps ymm7, ymm7, ymmword ptr [r10 + 3296]
    vmovups ymm8, ymmword ptr [r10 + 480]
    vaddps ymm8, ymm8, ymmword ptr [r10 + 1504]
    vaddps ymm8, ymm8, ymmword ptr [r10 + 2528]
    vaddps ymm8, ymm8, ymmword ptr [r10 + 3552]
    vmovups ymm9, ymmword ptr [r10 + 736]
    vaddps ymm9, ymm9, ymmword ptr [r10 + 1760]
    vaddps ymm9, ymm9, ymmword ptr [r10 + 2784]
    vaddps ymm9, ymm9, ymmword ptr [r10 + 3808]
    vmovups ymm10, ymmword ptr [r10 + 992]
    vaddps ymm10, ymm10, ymmword ptr [r10 + 2016]
    vaddps ymm10, ymm10, ymmword ptr [r10 + 3040]
    vaddps ymm10, ymm10, ymmword ptr [r10 + 4064]
    vaddps ymm7, ymm7, ymm9
    vaddps ymm8, ymm8, ymm10
    vaddps ymm7, ymm7, ymm8
    # the rest of the tree
    vaddps ymm0, ymm0, ymm4
    vaddps ymm1, ymm1, ymm5
    vaddps ymm2, ymm2, ymm6
    vaddps ymm3, ymm3, ymm7
    vaddps ymm0, ymm0, ymm2
    vaddps ymm1, ymm1, ymm3
    vaddps ymm0, ymm0, ymm1
    # the lanes
    vextractf128 xmm1, ymm0, 1
    vaddps xmm1, xmm0, xmm1
    vmovhlps xmm0, xmm1, xmm1
    vaddps xmm1, xmm1, xmm0
    vmovshdup xmm0, xmm1
    vaddss xmm1, xmm1, xmm0
    vmovss dword ptr [rsi + r9*4], xmm1
    add r9, qword ptr [rcx + 8]
    jmp 2b
6:
    # grid.sync()
    mov r9, qword ptr [rcx + 16]
    mov r10, qword ptr [r9 + 8]
    mov r11, 1
    lock xadd qword ptr [r9], r11
    inc r11
    cmp r11, qword ptr [rcx + 8]
    jnz 7f
    # the last to come lets the others go
    mov qword ptr [r9], 0
    inc r10
    mov qword ptr [r9 + 8], r10
    jmp 8f
7:
    # the others wait for it
    pause
    cmp r10, qword ptr [r9 + 8]
    jz 7b
8:
    # the chunks' sums, in order
    lea rdx, [rdx + 1023]
    shr rdx, 10
    vxorps xmm1, xmm1, xmm1
    xor r9d, r9d
9:
    cmp r9, rdx
    jz 12f
    vaddss xmm1, xmm1, dword ptr [rsi + r9*4]
    inc r9
    jmp 9b
12:
    # the thread's floats over the sum
    vbroadcastss ymm0, xmm1
    mov r9, qword ptr [rcx]
13:
    mov r10, r9
    shl r10, 12
    add r10, rdi
    cmp r9, rax
    jae 15f
    xor r11d, r11d
14:
    vmovups ymm2, ymmword ptr [r10 + r11*1]
    vdivps ymm2, ymm2, ymm0
    vmovups ymmword ptr [r10 + r11*1], ymm2
    vmovups ymm2, ymmword ptr [r10 + r11*1 + 32]
    vdivps ymm2, ymm2, ymm0
    vmovups ymmword ptr [r10 + r11*1 + 32], ymm2
    vmovups ymm2, ymmword ptr [r10 + r11*1 + 64]
    vdivps ymm2, ymm2, ymm0
    vmovups ymmword ptr [r10 + r11*1 + 64], ymm2
    vmovups ymm2, ymmword ptr [r10 + r11*1 + 96]
    vdivps ymm2, ymm2, ymm0
    vmovups ymmword ptr [r10 + r11*1 + 96], ymm2
    add r11, 128
    cmp r11, 4096
    jnz 14b
    add r9, qword ptr [rcx + 8]
    jmp 13b
15:
    jnz 17f
    test r8, r8
    jz 17f
    xor r11d, r11d
16:
    vmovss xmm0, dword ptr [r10 + r11*4]
    vdivss xmm0, xmm0, xmm1
    vmovss dword ptr [r10 + r11*4], xmm0
    inc r11
    cmp r11, r8
    jnz 16b
17:
    mov rsp, rbp
    pop rbp
    vzeroupper
    ret
