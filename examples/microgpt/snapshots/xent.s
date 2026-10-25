# xent(gpr, gpr, gpr, vec, gpr) -> ()
# frame: 0 bytes aligned to 0, saves []
# d=rdi probs=rsi targets=rdx inv_n=ymm0 n=rcx g=ymm1
# j=rax
    test rcx, rcx
    jz 3f
    vbroadcastss ymm0, xmm0
2:
    vmulps ymm1, ymm0, ymmword ptr [rsi]
    vmovups ymmword ptr [rdi], ymm1
    vmulps ymm1, ymm0, ymmword ptr [rsi + 32]
    vmovups ymmword ptr [rdi + 32], ymm1
    vmulps ymm1, ymm0, ymmword ptr [rsi + 64]
    vmovups ymmword ptr [rdi + 64], ymm1
    vmulps ymm1, ymm0, ymmword ptr [rsi + 96]
    vmovups ymmword ptr [rdi + 96], ymm1
    # the target's
    mov eax, dword ptr [rdx]
    vmovss xmm1, dword ptr [rsi + rax*4]
    vsubss xmm1, xmm1, dword ptr [rip + {c0}]
    vmulss xmm1, xmm1, xmm0
    vmovss dword ptr [rdi + rax*4], xmm1
    add rdi, 128
    add rsi, 128
    add rdx, 4
    dec rcx
    jnz 2b
3:
    vzeroupper
    ret
# c0, aligned to 4:
#   00 00 80 3f
