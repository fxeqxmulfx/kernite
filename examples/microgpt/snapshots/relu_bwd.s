# relu_bwd(gpr, gpr, gpr) -> ()
# frame: 0 bytes aligned to 0, saves []
# dh=rdi h=rsi n=rdx z=ymm0 keep=ymm1
    test rdx, rdx
    jz 3f
    vxorps xmm0, xmm0, xmm0
2:
    # 0 < h, which no NaN is
    vcmpps ymm1, ymm0, ymmword ptr [rsi], 0x11
    vandps ymm1, ymm1, ymmword ptr [rdi]
    vmovups ymmword ptr [rdi], ymm1
    add rsi, 32
    add rdi, 32
    dec rdx
    jnz 2b
3:
    vzeroupper
    ret
