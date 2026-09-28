# bf16_two(gpr, gpr, gpr, gpr, gpr, gpr) -> ()
# frame: 0 bytes aligned to 0, saves [rbx]
# w=rdi x=rsi out=rdx gap=rcx rows=r8 blocks=r9
# second=rax x0=r10 x1=r11 left=rbx acc00=ymm0 acc01=ymm1
# acc02=ymm2 acc03=ymm3 acc10=ymm4 acc11=ymm5 acc12=ymm6 acc13=ymm7
# w8=ymm8 s=ymm0 t=ymm1 high=ymm1 sum=ymm3 lane=ymm2
    push rbx
    # the second vector, 128 bytes a block past the first
    mov rax, r9
    shl rax, 7
    lea rax, [rsi + rax*1]
2:
    # a row
    vxorps xmm0, xmm0, xmm0
    vxorps xmm1, xmm1, xmm1
    vxorps xmm2, xmm2, xmm2
    vxorps xmm3, xmm3, xmm3
    vxorps xmm4, xmm4, xmm4
    vxorps xmm5, xmm5, xmm5
    vxorps xmm6, xmm6, xmm6
    vxorps xmm7, xmm7, xmm7
    mov r10, rsi
    mov r11, rax
    mov rbx, r9
3:
    # 32 values of the row
    vpmovzxwd ymm8, xmmword ptr [rdi]
    vpslld ymm8, ymm8, 16
    vfmadd231ps ymm0, ymm8, ymmword ptr [r10]
    vfmadd231ps ymm4, ymm8, ymmword ptr [r11]
    vpmovzxwd ymm8, xmmword ptr [rdi + 16]
    vpslld ymm8, ymm8, 16
    vfmadd231ps ymm1, ymm8, ymmword ptr [r10 + 32]
    vfmadd231ps ymm5, ymm8, ymmword ptr [r11 + 32]
    vpmovzxwd ymm8, xmmword ptr [rdi + 32]
    vpslld ymm8, ymm8, 16
    vfmadd231ps ymm2, ymm8, ymmword ptr [r10 + 64]
    vfmadd231ps ymm6, ymm8, ymmword ptr [r11 + 64]
    vpmovzxwd ymm8, xmmword ptr [rdi + 48]
    vpslld ymm8, ymm8, 16
    vfmadd231ps ymm3, ymm8, ymmword ptr [r10 + 96]
    vfmadd231ps ymm7, ymm8, ymmword ptr [r11 + 96]
    add rdi, 64
    add r10, 128
    add r11, 128
    dec rbx
    jnz 3b
    # the accumulators summed, then the lanes in order
    vaddps ymm0, ymm0, ymm1
    vaddps ymm1, ymm2, ymm3
    vaddps ymm0, ymm0, ymm1
    vextractf128 xmm1, ymm0, 1
    vmovshdup xmm2, xmm0
    vaddss xmm3, xmm0, xmm2
    vmovhlps xmm2, xmm0, xmm0
    vaddss xmm3, xmm3, xmm2
    vshufps xmm2, xmm0, xmm0, 0xff
    vaddss xmm3, xmm3, xmm2
    vaddss xmm3, xmm3, xmm1
    vmovshdup xmm2, xmm1
    vaddss xmm3, xmm3, xmm2
    vmovhlps xmm2, xmm1, xmm1
    vaddss xmm3, xmm3, xmm2
    vshufps xmm2, xmm1, xmm1, 0xff
    vaddss xmm3, xmm3, xmm2
    vmovss dword ptr [rdx], xmm3
    # the accumulators summed, then the lanes in order
    vaddps ymm0, ymm4, ymm5
    vaddps ymm1, ymm6, ymm7
    vaddps ymm0, ymm0, ymm1
    vextractf128 xmm1, ymm0, 1
    vmovshdup xmm2, xmm0
    vaddss xmm3, xmm0, xmm2
    vmovhlps xmm2, xmm0, xmm0
    vaddss xmm3, xmm3, xmm2
    vshufps xmm2, xmm0, xmm0, 0xff
    vaddss xmm3, xmm3, xmm2
    vaddss xmm3, xmm3, xmm1
    vmovshdup xmm2, xmm1
    vaddss xmm3, xmm3, xmm2
    vmovhlps xmm2, xmm1, xmm1
    vaddss xmm3, xmm3, xmm2
    vshufps xmm2, xmm1, xmm1, 0xff
    vaddss xmm3, xmm3, xmm2
    vmovss dword ptr [rdx + rcx*1], xmm3
    add rdx, 4
    dec r8
    jnz 2b
    pop rbx
    vzeroupper
    ret
