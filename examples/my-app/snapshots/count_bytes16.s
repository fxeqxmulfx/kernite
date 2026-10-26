# count_bytes::<16>(gpr, gpr, gpr) -> ()
# needs: avx, avx2
# frame: 16384 bytes aligned to 32, saves []
# tables: 16384 bytes at rsp
# bytes=rdi n=rsi counts=rdx z=ymm0 i=rax b=rax
# count=rcx s=ymm0
    push rbp
    mov rbp, rsp
    # the frame a page at a time, each touched, so that an overflow faults
    lea r11, [rsp - 16384]
2:
    sub rsp, 4096
    or qword ptr [rsp], 0
    cmp rsp, r11
    jne 2b
    and rsp, -32
    # the tables cleared
    vxorps xmm0, xmm0, xmm0
    xor eax, eax
3:
    vmovaps ymmword ptr [rsp + rax*1], ymm0
    vmovaps ymmword ptr [rsp + rax*1 + 32], ymm0
    vmovaps ymmword ptr [rsp + rax*1 + 64], ymm0
    vmovaps ymmword ptr [rsp + rax*1 + 96], ymm0
    add rax, 128
    cmp rax, 16384
    jnz 3b
    # a byte to each table in turn
    cmp rsi, 16
    jb 5f
4:
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
    movzx eax, byte ptr [rdi + 4]
    mov ecx, dword ptr [rsp + rax*4 + 4096]
    inc rcx
    mov dword ptr [rsp + rax*4 + 4096], ecx
    movzx eax, byte ptr [rdi + 5]
    mov ecx, dword ptr [rsp + rax*4 + 5120]
    inc rcx
    mov dword ptr [rsp + rax*4 + 5120], ecx
    movzx eax, byte ptr [rdi + 6]
    mov ecx, dword ptr [rsp + rax*4 + 6144]
    inc rcx
    mov dword ptr [rsp + rax*4 + 6144], ecx
    movzx eax, byte ptr [rdi + 7]
    mov ecx, dword ptr [rsp + rax*4 + 7168]
    inc rcx
    mov dword ptr [rsp + rax*4 + 7168], ecx
    movzx eax, byte ptr [rdi + 8]
    mov ecx, dword ptr [rsp + rax*4 + 8192]
    inc rcx
    mov dword ptr [rsp + rax*4 + 8192], ecx
    movzx eax, byte ptr [rdi + 9]
    mov ecx, dword ptr [rsp + rax*4 + 9216]
    inc rcx
    mov dword ptr [rsp + rax*4 + 9216], ecx
    movzx eax, byte ptr [rdi + 10]
    mov ecx, dword ptr [rsp + rax*4 + 10240]
    inc rcx
    mov dword ptr [rsp + rax*4 + 10240], ecx
    movzx eax, byte ptr [rdi + 11]
    mov ecx, dword ptr [rsp + rax*4 + 11264]
    inc rcx
    mov dword ptr [rsp + rax*4 + 11264], ecx
    movzx eax, byte ptr [rdi + 12]
    mov ecx, dword ptr [rsp + rax*4 + 12288]
    inc rcx
    mov dword ptr [rsp + rax*4 + 12288], ecx
    movzx eax, byte ptr [rdi + 13]
    mov ecx, dword ptr [rsp + rax*4 + 13312]
    inc rcx
    mov dword ptr [rsp + rax*4 + 13312], ecx
    movzx eax, byte ptr [rdi + 14]
    mov ecx, dword ptr [rsp + rax*4 + 14336]
    inc rcx
    mov dword ptr [rsp + rax*4 + 14336], ecx
    movzx eax, byte ptr [rdi + 15]
    mov ecx, dword ptr [rsp + rax*4 + 15360]
    inc rcx
    mov dword ptr [rsp + rax*4 + 15360], ecx
    add rdi, 16
    sub rsi, 16
    cmp rsi, 16
    jae 4b
5:
    test rsi, rsi
    jz 7f
6:
    movzx eax, byte ptr [rdi]
    mov ecx, dword ptr [rsp + rax*4]
    inc rcx
    mov dword ptr [rsp + rax*4], ecx
    inc rdi
    dec rsi
    jnz 6b
7:
    # the tables added to the counts
    xor eax, eax
8:
    vmovdqa ymm0, ymmword ptr [rsp + rax*1]
    vpaddd ymm0, ymm0, ymmword ptr [rsp + rax*1 + 1024]
    vpaddd ymm0, ymm0, ymmword ptr [rsp + rax*1 + 2048]
    vpaddd ymm0, ymm0, ymmword ptr [rsp + rax*1 + 3072]
    vpaddd ymm0, ymm0, ymmword ptr [rsp + rax*1 + 4096]
    vpaddd ymm0, ymm0, ymmword ptr [rsp + rax*1 + 5120]
    vpaddd ymm0, ymm0, ymmword ptr [rsp + rax*1 + 6144]
    vpaddd ymm0, ymm0, ymmword ptr [rsp + rax*1 + 7168]
    vpaddd ymm0, ymm0, ymmword ptr [rsp + rax*1 + 8192]
    vpaddd ymm0, ymm0, ymmword ptr [rsp + rax*1 + 9216]
    vpaddd ymm0, ymm0, ymmword ptr [rsp + rax*1 + 10240]
    vpaddd ymm0, ymm0, ymmword ptr [rsp + rax*1 + 11264]
    vpaddd ymm0, ymm0, ymmword ptr [rsp + rax*1 + 12288]
    vpaddd ymm0, ymm0, ymmword ptr [rsp + rax*1 + 13312]
    vpaddd ymm0, ymm0, ymmword ptr [rsp + rax*1 + 14336]
    vpaddd ymm0, ymm0, ymmword ptr [rsp + rax*1 + 15360]
    vpaddd ymm0, ymm0, ymmword ptr [rdx + rax*1]
    vmovdqu ymmword ptr [rdx + rax*1], ymm0
    add rax, 32
    cmp rax, 1024
    jnz 8b
    mov rsp, rbp
    pop rbp
    vzeroupper
    ret
