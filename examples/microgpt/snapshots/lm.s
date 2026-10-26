# lm(gpr, gpr, gpr, gpr) -> ()
# needs: avx, fma
# frame: 0 bytes aligned to 0, saves []
# y=rdi x=rsi w=rdx n=rcx acc0=ymm0 acc1=ymm1
# acc2=ymm2 acc3=ymm3 b=ymm4
    test rcx, rcx
    jz 3f
2:
    vxorps xmm0, xmm0, xmm0
    vxorps xmm1, xmm1, xmm1
    vxorps xmm2, xmm2, xmm2
    vxorps xmm3, xmm3, xmm3
    vbroadcastss ymm4, dword ptr [rsi]
    vfmadd231ps ymm0, ymm4, ymmword ptr [rdx]
    vfmadd231ps ymm1, ymm4, ymmword ptr [rdx + 32]
    vfmadd231ps ymm2, ymm4, ymmword ptr [rdx + 64]
    vfmadd231ps ymm3, ymm4, ymmword ptr [rdx + 96]
    vbroadcastss ymm4, dword ptr [rsi + 4]
    vfmadd231ps ymm0, ymm4, ymmword ptr [rdx + 128]
    vfmadd231ps ymm1, ymm4, ymmword ptr [rdx + 160]
    vfmadd231ps ymm2, ymm4, ymmword ptr [rdx + 192]
    vfmadd231ps ymm3, ymm4, ymmword ptr [rdx + 224]
    vbroadcastss ymm4, dword ptr [rsi + 8]
    vfmadd231ps ymm0, ymm4, ymmword ptr [rdx + 256]
    vfmadd231ps ymm1, ymm4, ymmword ptr [rdx + 288]
    vfmadd231ps ymm2, ymm4, ymmword ptr [rdx + 320]
    vfmadd231ps ymm3, ymm4, ymmword ptr [rdx + 352]
    vbroadcastss ymm4, dword ptr [rsi + 12]
    vfmadd231ps ymm0, ymm4, ymmword ptr [rdx + 384]
    vfmadd231ps ymm1, ymm4, ymmword ptr [rdx + 416]
    vfmadd231ps ymm2, ymm4, ymmword ptr [rdx + 448]
    vfmadd231ps ymm3, ymm4, ymmword ptr [rdx + 480]
    vbroadcastss ymm4, dword ptr [rsi + 16]
    vfmadd231ps ymm0, ymm4, ymmword ptr [rdx + 512]
    vfmadd231ps ymm1, ymm4, ymmword ptr [rdx + 544]
    vfmadd231ps ymm2, ymm4, ymmword ptr [rdx + 576]
    vfmadd231ps ymm3, ymm4, ymmword ptr [rdx + 608]
    vbroadcastss ymm4, dword ptr [rsi + 20]
    vfmadd231ps ymm0, ymm4, ymmword ptr [rdx + 640]
    vfmadd231ps ymm1, ymm4, ymmword ptr [rdx + 672]
    vfmadd231ps ymm2, ymm4, ymmword ptr [rdx + 704]
    vfmadd231ps ymm3, ymm4, ymmword ptr [rdx + 736]
    vbroadcastss ymm4, dword ptr [rsi + 24]
    vfmadd231ps ymm0, ymm4, ymmword ptr [rdx + 768]
    vfmadd231ps ymm1, ymm4, ymmword ptr [rdx + 800]
    vfmadd231ps ymm2, ymm4, ymmword ptr [rdx + 832]
    vfmadd231ps ymm3, ymm4, ymmword ptr [rdx + 864]
    vbroadcastss ymm4, dword ptr [rsi + 28]
    vfmadd231ps ymm0, ymm4, ymmword ptr [rdx + 896]
    vfmadd231ps ymm1, ymm4, ymmword ptr [rdx + 928]
    vfmadd231ps ymm2, ymm4, ymmword ptr [rdx + 960]
    vfmadd231ps ymm3, ymm4, ymmword ptr [rdx + 992]
    vbroadcastss ymm4, dword ptr [rsi + 32]
    vfmadd231ps ymm0, ymm4, ymmword ptr [rdx + 1024]
    vfmadd231ps ymm1, ymm4, ymmword ptr [rdx + 1056]
    vfmadd231ps ymm2, ymm4, ymmword ptr [rdx + 1088]
    vfmadd231ps ymm3, ymm4, ymmword ptr [rdx + 1120]
    vbroadcastss ymm4, dword ptr [rsi + 36]
    vfmadd231ps ymm0, ymm4, ymmword ptr [rdx + 1152]
    vfmadd231ps ymm1, ymm4, ymmword ptr [rdx + 1184]
    vfmadd231ps ymm2, ymm4, ymmword ptr [rdx + 1216]
    vfmadd231ps ymm3, ymm4, ymmword ptr [rdx + 1248]
    vbroadcastss ymm4, dword ptr [rsi + 40]
    vfmadd231ps ymm0, ymm4, ymmword ptr [rdx + 1280]
    vfmadd231ps ymm1, ymm4, ymmword ptr [rdx + 1312]
    vfmadd231ps ymm2, ymm4, ymmword ptr [rdx + 1344]
    vfmadd231ps ymm3, ymm4, ymmword ptr [rdx + 1376]
    vbroadcastss ymm4, dword ptr [rsi + 44]
    vfmadd231ps ymm0, ymm4, ymmword ptr [rdx + 1408]
    vfmadd231ps ymm1, ymm4, ymmword ptr [rdx + 1440]
    vfmadd231ps ymm2, ymm4, ymmword ptr [rdx + 1472]
    vfmadd231ps ymm3, ymm4, ymmword ptr [rdx + 1504]
    vbroadcastss ymm4, dword ptr [rsi + 48]
    vfmadd231ps ymm0, ymm4, ymmword ptr [rdx + 1536]
    vfmadd231ps ymm1, ymm4, ymmword ptr [rdx + 1568]
    vfmadd231ps ymm2, ymm4, ymmword ptr [rdx + 1600]
    vfmadd231ps ymm3, ymm4, ymmword ptr [rdx + 1632]
    vbroadcastss ymm4, dword ptr [rsi + 52]
    vfmadd231ps ymm0, ymm4, ymmword ptr [rdx + 1664]
    vfmadd231ps ymm1, ymm4, ymmword ptr [rdx + 1696]
    vfmadd231ps ymm2, ymm4, ymmword ptr [rdx + 1728]
    vfmadd231ps ymm3, ymm4, ymmword ptr [rdx + 1760]
    vbroadcastss ymm4, dword ptr [rsi + 56]
    vfmadd231ps ymm0, ymm4, ymmword ptr [rdx + 1792]
    vfmadd231ps ymm1, ymm4, ymmword ptr [rdx + 1824]
    vfmadd231ps ymm2, ymm4, ymmword ptr [rdx + 1856]
    vfmadd231ps ymm3, ymm4, ymmword ptr [rdx + 1888]
    vbroadcastss ymm4, dword ptr [rsi + 60]
    vfmadd231ps ymm0, ymm4, ymmword ptr [rdx + 1920]
    vfmadd231ps ymm1, ymm4, ymmword ptr [rdx + 1952]
    vfmadd231ps ymm2, ymm4, ymmword ptr [rdx + 1984]
    vfmadd231ps ymm3, ymm4, ymmword ptr [rdx + 2016]
    vmovups ymmword ptr [rdi], ymm0
    vmovups ymmword ptr [rdi + 32], ymm1
    vmovups ymmword ptr [rdi + 64], ymm2
    vmovups ymmword ptr [rdi + 96], ymm3
    add rsi, 64
    add rdi, 128
    dec rcx
    jnz 2b
3:
    vzeroupper
    ret
