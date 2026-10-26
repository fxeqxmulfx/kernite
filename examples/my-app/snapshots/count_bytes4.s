# count_bytes::<4>(gpr, gpr, gpr) -> ()
# needs: avx, avx2
# frame: 4096 bytes aligned to 32, saves []
# tables: 4096 bytes at rsp
# bytes=rdi n=rsi counts=rdx z=ymm0 i=rax b=rax
# count=rcx s=ymm0
    push rbp
    mov rbp, rsp
    # the frame a page at a time, each touched, so that an overflow faults
    sub rsp, 4096
    or qword ptr [rsp], 0
    and rsp, -32
    # the tables cleared
    vxorps xmm0, xmm0, xmm0
    xor eax, eax
2:
    vmovaps ymmword ptr [rsp + rax*1], ymm0
    vmovaps ymmword ptr [rsp + rax*1 + 32], ymm0
    vmovaps ymmword ptr [rsp + rax*1 + 64], ymm0
    vmovaps ymmword ptr [rsp + rax*1 + 96], ymm0
    add rax, 128
    cmp rax, 4096
    jnz 2b
    # a byte to each table in turn
    cmp rsi, 4
    jb 4f
3:
    movzx eax, byte ptr [rdi]
    mov ecx, dword ptr [rsp + rax*4]
    inc rcx
    mov dword ptr [rsp + rax*4], ecx
    movzx eax, byte ptr [rdi + 1]
    mov ecx, dword ptr [rsp + rax*4 + 1024]
    inc rcx
    mov dword ptr [rsp + rax*4 + 1024], ecx
    movzx eax, byte ptr [rdi + 2]
    mov ecx, dword ptr [rsp + rax*4 + 2048]
    inc rcx
    mov dword ptr [rsp + rax*4 + 2048], ecx
    movzx eax, byte ptr [rdi + 3]
    mov ecx, dword ptr [rsp + rax*4 + 3072]
    inc rcx
    mov dword ptr [rsp + rax*4 + 3072], ecx
    add rdi, 4
    sub rsi, 4
    cmp rsi, 4
    jae 3b
4:
    test rsi, rsi
    jz 6f
5:
    movzx eax, byte ptr [rdi]
    mov ecx, dword ptr [rsp + rax*4]
    inc rcx
    mov dword ptr [rsp + rax*4], ecx
    inc rdi
    dec rsi
    jnz 5b
6:
    # the tables added to the counts
    xor eax, eax
7:
    vmovdqa ymm0, ymmword ptr [rsp + rax*1]
    vpaddd ymm0, ymm0, ymmword ptr [rsp + rax*1 + 1024]
    vpaddd ymm0, ymm0, ymmword ptr [rsp + rax*1 + 2048]
    vpaddd ymm0, ymm0, ymmword ptr [rsp + rax*1 + 3072]
    vpaddd ymm0, ymm0, ymmword ptr [rdx + rax*1]
    vmovdqu ymmword ptr [rdx + rax*1], ymm0
    add rax, 32
    cmp rax, 1024
    jnz 7b
    mov rsp, rbp
    pop rbp
    vzeroupper
    ret
