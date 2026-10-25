# adam(gpr, gpr, gpr, gpr, gpr, vec, vec, vec, vec, vec, vec, vec, vec) -> ()
# frame: 0 bytes aligned to 0, saves []
# p=rdi g=rsi m=rdx v=rcx n=r8 lr=ymm0
# b1=ymm1 ob1=ymm2 b2=ymm3 ob2=ymm4 bc1=ymm5 bc2=ymm6
# eps=ymm7 z=ymm8 gs=ymm9 ms=ymm10 vs=ymm11 ps=ymm9
    test r8, r8
    jz 3f
    vbroadcastss ymm0, xmm0
    vbroadcastss ymm1, xmm1
    vbroadcastss ymm2, xmm2
    vbroadcastss ymm3, xmm3
    vbroadcastss ymm4, xmm4
    vbroadcastss ymm5, xmm5
    vbroadcastss ymm6, xmm6
    vbroadcastss ymm7, xmm7
    vxorps xmm8, xmm8, xmm8
2:
    vmovups ymm9, ymmword ptr [rsi]
    # m = b1 m + (1 - b1) g; v = b2 v + (1 - b2) g^2
    vmulps ymm10, ymm1, ymmword ptr [rdx]
    vfmadd231ps ymm10, ymm2, ymm9
    vmovups ymmword ptr [rdx], ymm10
    vmulps ymm9, ymm9, ymm9
    vmulps ymm11, ymm3, ymmword ptr [rcx]
    vfmadd231ps ymm11, ymm4, ymm9
    vmovups ymmword ptr [rcx], ymm11
    # p -= lr m^ / (sqrt(v^) + eps)
    vmulps ymm10, ymm10, ymm5
    vmulps ymm11, ymm11, ymm6
    vsqrtps ymm11, ymm11
    vaddps ymm11, ymm11, ymm7
    vmulps ymm10, ymm0, ymm10
    vdivps ymm10, ymm10, ymm11
    vmovups ymm9, ymmword ptr [rdi]
    vsubps ymm9, ymm9, ymm10
    vmovups ymmword ptr [rdi], ymm9
    vmovups ymmword ptr [rsi], ymm8
    add rdi, 32
    add rsi, 32
    add rdx, 32
    add rcx, 32
    dec r8
    jnz 2b
3:
    vzeroupper
    ret
