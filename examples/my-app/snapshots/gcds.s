# gcds(gpr, gpr, gpr, gpr) -> ()
# frame: 0 bytes aligned to 0, saves [rbx, r12, r13, r14, r15]
# a=r14,rdi b=r13,rsi g=r12,rdx n=rbx,rcx i=r15 x=rdi
# y=rsi d=rax
    push rbx
    push r12
    push r13
    push r14
    push r15
    mov rbx, rcx
    mov r12, rdx
    mov r13, rsi
    mov r14, rdi
    test rbx, rbx
    jz 3f
    xor r15d, r15d
2:
    mov edi, dword ptr [r14 + r15*4]
    mov esi, dword ptr [r13 + r15*4]
    call {f0}
    mov dword ptr [r12 + r15*4], eax
    inc r15
    cmp r15, rbx
    jnz 2b
3:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
# f0: gcd
