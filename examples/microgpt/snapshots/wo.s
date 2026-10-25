# wo(gpr, gpr, gpr, gpr, gpr) -> ()
# frame: 0 bytes aligned to 0, saves []
# y=rdi x=rsi w=rdx res=rcx n=r8 acc0=ymm0
# acc1=ymm1 b=ymm2
    test r8, r8
    jz 3f
2:
    vxorps xmm0, xmm0, xmm0
    vxorps xmm1, xmm1, xmm1
    vbroadcastss ymm2, dword ptr [rsi]
    vfmadd231ps ymm0, ymm2, ymmword ptr [rdx]
    vfmadd231ps ymm1, ymm2, ymmword ptr [rdx + 32]
    vbroadcastss ymm2, dword ptr [rsi + 4]
    vfmadd231ps ymm0, ymm2, ymmword ptr [rdx + 64]
    vfmadd231ps ymm1, ymm2, ymmword ptr [rdx + 96]
    vbroadcastss ymm2, dword ptr [rsi + 8]
    vfmadd231ps ymm0, ymm2, ymmword ptr [rdx + 128]
    vfmadd231ps ymm1, ymm2, ymmword ptr [rdx + 160]
    vbroadcastss ymm2, dword ptr [rsi + 12]
    vfmadd231ps ymm0, ymm2, ymmword ptr [rdx + 192]
    vfmadd231ps ymm1, ymm2, ymmword ptr [rdx + 224]
    vbroadcastss ymm2, dword ptr [rsi + 16]
    vfmadd231ps ymm0, ymm2, ymmword ptr [rdx + 256]
    vfmadd231ps ymm1, ymm2, ymmword ptr [rdx + 288]
    vbroadcastss ymm2, dword ptr [rsi + 20]
    vfmadd231ps ymm0, ymm2, ymmword ptr [rdx + 320]
    vfmadd231ps ymm1, ymm2, ymmword ptr [rdx + 352]
    vbroadcastss ymm2, dword ptr [rsi + 24]
    vfmadd231ps ymm0, ymm2, ymmword ptr [rdx + 384]
    vfmadd231ps ymm1, ymm2, ymmword ptr [rdx + 416]
    vbroadcastss ymm2, dword ptr [rsi + 28]
    vfmadd231ps ymm0, ymm2, ymmword ptr [rdx + 448]
    vfmadd231ps ymm1, ymm2, ymmword ptr [rdx + 480]
    vbroadcastss ymm2, dword ptr [rsi + 32]
    vfmadd231ps ymm0, ymm2, ymmword ptr [rdx + 512]
    vfmadd231ps ymm1, ymm2, ymmword ptr [rdx + 544]
    vbroadcastss ymm2, dword ptr [rsi + 36]
    vfmadd231ps ymm0, ymm2, ymmword ptr [rdx + 576]
    vfmadd231ps ymm1, ymm2, ymmword ptr [rdx + 608]
    vbroadcastss ymm2, dword ptr [rsi + 40]
    vfmadd231ps ymm0, ymm2, ymmword ptr [rdx + 640]
    vfmadd231ps ymm1, ymm2, ymmword ptr [rdx + 672]
    vbroadcastss ymm2, dword ptr [rsi + 44]
    vfmadd231ps ymm0, ymm2, ymmword ptr [rdx + 704]
    vfmadd231ps ymm1, ymm2, ymmword ptr [rdx + 736]
    vbroadcastss ymm2, dword ptr [rsi + 48]
    vfmadd231ps ymm0, ymm2, ymmword ptr [rdx + 768]
    vfmadd231ps ymm1, ymm2, ymmword ptr [rdx + 800]
    vbroadcastss ymm2, dword ptr [rsi + 52]
    vfmadd231ps ymm0, ymm2, ymmword ptr [rdx + 832]
    vfmadd231ps ymm1, ymm2, ymmword ptr [rdx + 864]
    vbroadcastss ymm2, dword ptr [rsi + 56]
    vfmadd231ps ymm0, ymm2, ymmword ptr [rdx + 896]
    vfmadd231ps ymm1, ymm2, ymmword ptr [rdx + 928]
    vbroadcastss ymm2, dword ptr [rsi + 60]
    vfmadd231ps ymm0, ymm2, ymmword ptr [rdx + 960]
    vfmadd231ps ymm1, ymm2, ymmword ptr [rdx + 992]
    vaddps ymm0, ymm0, ymmword ptr [rcx]
    vmovups ymmword ptr [rdi], ymm0
    vaddps ymm1, ymm1, ymmword ptr [rcx + 32]
    vmovups ymmword ptr [rdi + 32], ymm1
    add rsi, 64
    add rdi, 64
    add rcx, 64
    dec r8
    jnz 2b
3:
    vzeroupper
    ret
