# gcd(gpr, gpr) -> gpr
# frame: 0 bytes aligned to 0, saves [], and 8 more so that it calls at a multiple of 16
# a=rdi,rax b=rsi t=rax g=rax
    sub rsp, 8
    # `a` the larger
    cmp rdi, rsi
    jae 2f
    mov rax, rdi
    mov rdi, rsi
    mov rsi, rax
2:
    test rsi, rsi
    jnz 3f
    mov rax, rdi
    add rsp, 8
    ret
3:
    test rdi, 1
    jnz 4f
    test rsi, 1
    jnz 5f
    # both even: twice the gcd of their halves
    shr rdi, 1
    shr rsi, 1
    call {f0}
    add rax, rax
    add rsp, 8
    ret
4:
    test rsi, 1
    jz 6f
    # both odd: their difference, which is even, for `a`
    sub rdi, rsi
5:
    # `a` even and `b` odd: `a` halved
    shr rdi, 1
    call {f0}
    add rsp, 8
    ret
6:
    # `b` even and `a` odd: `b` halved
    shr rsi, 1
    call {f0}
    add rsp, 8
    ret
# f0: gcd
