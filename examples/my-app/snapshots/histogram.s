# histogram(gpr, gpr) -> ()
# frame: 0 bytes aligned to 0, saves []
# x=rdi n=rsi bins=rax bin=rcx count=rdx
    test rsi, rsi
    jz 3f
    # the bins' address, for an index
    lea rax, [rip + {COUNTS}]
2:
    mov ecx, dword ptr [rdi]
    shr rcx, 28
    mov edx, dword ptr [rax + rcx*4]
    inc rdx
    mov dword ptr [rax + rcx*4], edx
    add rdi, 4
    dec rsi
    jnz 2b
3:
    ret
