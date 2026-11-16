# gcds(gpr, gpr, gpr, gpr) -> ()
# frame: 0 bytes aligned to 0, saves [], and 8 more so that it calls at a multiple of 16
# a=r9,rdi b=r8,rsi g=rdx n=rcx i=r10 x=rdi
# y=rsi d=rax
    sub rsp, 8
    mov r8, rsi
    mov r9, rdi
    test rcx, rcx
    jz 3f
    xor r10d, r10d
2:
    mov edi, dword ptr [r9 + r10*4]
    mov esi, dword ptr [r8 + r10*4]
    call {f0}
    mov dword ptr [rdx + r10*4], eax
    inc r10
    cmp r10, rcx
    jnz 2b
3:
    add rsp, 8
    ret
# f0: gcd
