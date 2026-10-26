# weigh(gpr, gpr) -> ()
# frame: 0 bytes aligned to 0, saves []
# x=rdi n=rsi w=ymm0 v=ymm1 weights=rax i=rcx
    vbroadcastf128 ymm0, xmmword ptr [rip + {WEIGHTS}]
    cmp rsi, 8
    jb 3f
2:
    vmulps ymm1, ymm0, ymmword ptr [rdi]
    vmovups ymmword ptr [rdi], ymm1
    add rdi, 32
    sub rsi, 8
    cmp rsi, 8
    jae 2b
3:
    cmp rsi, 4
    jb 4f
    vmulps xmm1, xmm0, xmmword ptr [rdi]
    vmovups xmmword ptr [rdi], xmm1
    add rdi, 16
    sub rsi, 4
4:
    test rsi, rsi
    jz 6f
    # the weights' address, for an index
    lea rax, [rip + {WEIGHTS}]
    xor ecx, ecx
5:
    vmovss xmm1, dword ptr [rdi + rcx*4]
    vmulss xmm1, xmm1, dword ptr [rax + rcx*4]
    vmovss dword ptr [rdi + rcx*4], xmm1
    inc rcx
    cmp rcx, rsi
    jnz 5b
6:
    vzeroupper
    ret
