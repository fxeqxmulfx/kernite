# gcds(gpr, gpr, gpr, gpr, gpr, gpr) -> ()
# frame: 0 bytes aligned to 0, saves [], and 8 more so that it calls at a multiple of 16
# a=r10,rdi a.len=rsi b=rdx b.len=rcx g=r8 g.len=r9
# n=r11 i=rcx x=rdi y=rsi d=rax
    sub rsp, 8
    mov r10, rdi
    mov r11, rsi
    cmp r11, rcx
    cmova r11, rcx
    cmp r11, r9
    cmova r11, r9
    test r11, r11
    jz 3f
    xor ecx, ecx
2:
    mov edi, dword ptr [r10 + rcx*4]
    mov esi, dword ptr [rdx + rcx*4]
    call {f0}
    mov dword ptr [r8 + rcx*4], eax
    inc rcx
    cmp rcx, r11
    jnz 2b
3:
    add rsp, 8
    ret
# f0: gcd
