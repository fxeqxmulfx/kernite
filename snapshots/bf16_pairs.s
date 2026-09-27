# bf16_pairs(gpr, gpr, gpr, gpr, gpr, gpr) -> ()
# frame: 0 bytes aligned to 0, saves [rbx]
# w=rdi x=rsi out=rdx pairs=rcx stride=r8 blocks=r9
# row0=rax row1=r10 xs=r11 left=rbx acc00=ymm0 acc01=ymm1
# acc02=ymm2 acc03=ymm3 acc10=ymm4 acc11=ymm5 acc12=ymm6 acc13=ymm7
# x8=ymm8 w0=ymm9 w1=ymm9 s=ymm0 t=ymm1 high=ymm1
# sum=ymm3 lane=ymm2
    push rbx
2:
    # a pair of rows
    vxorps xmm0, xmm0, xmm0
    vxorps xmm1, xmm1, xmm1
    vxorps xmm2, xmm2, xmm2
    vxorps xmm3, xmm3, xmm3
    vxorps xmm4, xmm4, xmm4
    vxorps xmm5, xmm5, xmm5
    vxorps xmm6, xmm6, xmm6
    vxorps xmm7, xmm7, xmm7
    mov rax, rdi
    lea r10, [rdi + r8*1]
    mov r11, rsi
    mov rbx, r9
3:
    # 32 values of each row
    vmovups ymm8, ymmword ptr [r11]
    vpmovzxwd ymm9, xmmword ptr [rax]
    vpslld ymm9, ymm9, 16
    vfmadd231ps ymm0, ymm9, ymm8
    vpmovzxwd ymm9, xmmword ptr [r10]
    vpslld ymm9, ymm9, 16
    vfmadd231ps ymm4, ymm9, ymm8
    vmovups ymm8, ymmword ptr [r11 + 32]
    vpmovzxwd ymm9, xmmword ptr [rax + 16]
    vpslld ymm9, ymm9, 16
    vfmadd231ps ymm1, ymm9, ymm8
    vpmovzxwd ymm9, xmmword ptr [r10 + 16]
    vpslld ymm9, ymm9, 16
    vfmadd231ps ymm5, ymm9, ymm8
    vmovups ymm8, ymmword ptr [r11 + 64]
    vpmovzxwd ymm9, xmmword ptr [rax + 32]
    vpslld ymm9, ymm9, 16
    vfmadd231ps ymm2, ymm9, ymm8
    vpmovzxwd ymm9, xmmword ptr [r10 + 32]
    vpslld ymm9, ymm9, 16
    vfmadd231ps ymm6, ymm9, ymm8
    vmovups ymm8, ymmword ptr [r11 + 96]
    vpmovzxwd ymm9, xmmword ptr [rax + 48]
    vpslld ymm9, ymm9, 16
    vfmadd231ps ymm3, ymm9, ymm8
    vpmovzxwd ymm9, xmmword ptr [r10 + 48]
    vpslld ymm9, ymm9, 16
    vfmadd231ps ymm7, ymm9, ymm8
    add rax, 64
    add r10, 64
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
    vmovss dword ptr [rdx + 4], xmm3
    add rdx, 8
    lea rdi, [rdi + r8*2]
    dec rcx
    jnz 2b
    pop rbx
    vzeroupper
    ret
