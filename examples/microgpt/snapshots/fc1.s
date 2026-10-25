# fc1(gpr, gpr, gpr, gpr) -> ()
# frame: 0 bytes aligned to 0, saves []
# y=rdi x=rsi w=rdx n=rcx z=ymm0 acc0=ymm1
# acc1=ymm2 acc2=ymm3 acc3=ymm4 acc4=ymm5 acc5=ymm6 acc6=ymm7
# acc7=ymm8 b=ymm9
    test rcx, rcx
    jz 3f
    vxorps xmm0, xmm0, xmm0
2:
    vxorps xmm1, xmm1, xmm1
    vxorps xmm2, xmm2, xmm2
    vxorps xmm3, xmm3, xmm3
    vxorps xmm4, xmm4, xmm4
    vxorps xmm5, xmm5, xmm5
    vxorps xmm6, xmm6, xmm6
    vxorps xmm7, xmm7, xmm7
    vxorps xmm8, xmm8, xmm8
    vbroadcastss ymm9, dword ptr [rsi]
    vfmadd231ps ymm1, ymm9, ymmword ptr [rdx]
    vfmadd231ps ymm2, ymm9, ymmword ptr [rdx + 32]
    vfmadd231ps ymm3, ymm9, ymmword ptr [rdx + 64]
    vfmadd231ps ymm4, ymm9, ymmword ptr [rdx + 96]
    vfmadd231ps ymm5, ymm9, ymmword ptr [rdx + 128]
    vfmadd231ps ymm6, ymm9, ymmword ptr [rdx + 160]
    vfmadd231ps ymm7, ymm9, ymmword ptr [rdx + 192]
    vfmadd231ps ymm8, ymm9, ymmword ptr [rdx + 224]
    vbroadcastss ymm9, dword ptr [rsi + 4]
    vfmadd231ps ymm1, ymm9, ymmword ptr [rdx + 256]
    vfmadd231ps ymm2, ymm9, ymmword ptr [rdx + 288]
    vfmadd231ps ymm3, ymm9, ymmword ptr [rdx + 320]
    vfmadd231ps ymm4, ymm9, ymmword ptr [rdx + 352]
    vfmadd231ps ymm5, ymm9, ymmword ptr [rdx + 384]
    vfmadd231ps ymm6, ymm9, ymmword ptr [rdx + 416]
    vfmadd231ps ymm7, ymm9, ymmword ptr [rdx + 448]
    vfmadd231ps ymm8, ymm9, ymmword ptr [rdx + 480]
    vbroadcastss ymm9, dword ptr [rsi + 8]
    vfmadd231ps ymm1, ymm9, ymmword ptr [rdx + 512]
    vfmadd231ps ymm2, ymm9, ymmword ptr [rdx + 544]
    vfmadd231ps ymm3, ymm9, ymmword ptr [rdx + 576]
    vfmadd231ps ymm4, ymm9, ymmword ptr [rdx + 608]
    vfmadd231ps ymm5, ymm9, ymmword ptr [rdx + 640]
    vfmadd231ps ymm6, ymm9, ymmword ptr [rdx + 672]
    vfmadd231ps ymm7, ymm9, ymmword ptr [rdx + 704]
    vfmadd231ps ymm8, ymm9, ymmword ptr [rdx + 736]
    vbroadcastss ymm9, dword ptr [rsi + 12]
    vfmadd231ps ymm1, ymm9, ymmword ptr [rdx + 768]
    vfmadd231ps ymm2, ymm9, ymmword ptr [rdx + 800]
    vfmadd231ps ymm3, ymm9, ymmword ptr [rdx + 832]
    vfmadd231ps ymm4, ymm9, ymmword ptr [rdx + 864]
    vfmadd231ps ymm5, ymm9, ymmword ptr [rdx + 896]
    vfmadd231ps ymm6, ymm9, ymmword ptr [rdx + 928]
    vfmadd231ps ymm7, ymm9, ymmword ptr [rdx + 960]
    vfmadd231ps ymm8, ymm9, ymmword ptr [rdx + 992]
    vbroadcastss ymm9, dword ptr [rsi + 16]
    vfmadd231ps ymm1, ymm9, ymmword ptr [rdx + 1024]
    vfmadd231ps ymm2, ymm9, ymmword ptr [rdx + 1056]
    vfmadd231ps ymm3, ymm9, ymmword ptr [rdx + 1088]
    vfmadd231ps ymm4, ymm9, ymmword ptr [rdx + 1120]
    vfmadd231ps ymm5, ymm9, ymmword ptr [rdx + 1152]
    vfmadd231ps ymm6, ymm9, ymmword ptr [rdx + 1184]
    vfmadd231ps ymm7, ymm9, ymmword ptr [rdx + 1216]
    vfmadd231ps ymm8, ymm9, ymmword ptr [rdx + 1248]
    vbroadcastss ymm9, dword ptr [rsi + 20]
    vfmadd231ps ymm1, ymm9, ymmword ptr [rdx + 1280]
    vfmadd231ps ymm2, ymm9, ymmword ptr [rdx + 1312]
    vfmadd231ps ymm3, ymm9, ymmword ptr [rdx + 1344]
    vfmadd231ps ymm4, ymm9, ymmword ptr [rdx + 1376]
    vfmadd231ps ymm5, ymm9, ymmword ptr [rdx + 1408]
    vfmadd231ps ymm6, ymm9, ymmword ptr [rdx + 1440]
    vfmadd231ps ymm7, ymm9, ymmword ptr [rdx + 1472]
    vfmadd231ps ymm8, ymm9, ymmword ptr [rdx + 1504]
    vbroadcastss ymm9, dword ptr [rsi + 24]
    vfmadd231ps ymm1, ymm9, ymmword ptr [rdx + 1536]
    vfmadd231ps ymm2, ymm9, ymmword ptr [rdx + 1568]
    vfmadd231ps ymm3, ymm9, ymmword ptr [rdx + 1600]
    vfmadd231ps ymm4, ymm9, ymmword ptr [rdx + 1632]
    vfmadd231ps ymm5, ymm9, ymmword ptr [rdx + 1664]
    vfmadd231ps ymm6, ymm9, ymmword ptr [rdx + 1696]
    vfmadd231ps ymm7, ymm9, ymmword ptr [rdx + 1728]
    vfmadd231ps ymm8, ymm9, ymmword ptr [rdx + 1760]
    vbroadcastss ymm9, dword ptr [rsi + 28]
    vfmadd231ps ymm1, ymm9, ymmword ptr [rdx + 1792]
    vfmadd231ps ymm2, ymm9, ymmword ptr [rdx + 1824]
    vfmadd231ps ymm3, ymm9, ymmword ptr [rdx + 1856]
    vfmadd231ps ymm4, ymm9, ymmword ptr [rdx + 1888]
    vfmadd231ps ymm5, ymm9, ymmword ptr [rdx + 1920]
    vfmadd231ps ymm6, ymm9, ymmword ptr [rdx + 1952]
    vfmadd231ps ymm7, ymm9, ymmword ptr [rdx + 1984]
    vfmadd231ps ymm8, ymm9, ymmword ptr [rdx + 2016]
    vbroadcastss ymm9, dword ptr [rsi + 32]
    vfmadd231ps ymm1, ymm9, ymmword ptr [rdx + 2048]
    vfmadd231ps ymm2, ymm9, ymmword ptr [rdx + 2080]
    vfmadd231ps ymm3, ymm9, ymmword ptr [rdx + 2112]
    vfmadd231ps ymm4, ymm9, ymmword ptr [rdx + 2144]
    vfmadd231ps ymm5, ymm9, ymmword ptr [rdx + 2176]
    vfmadd231ps ymm6, ymm9, ymmword ptr [rdx + 2208]
    vfmadd231ps ymm7, ymm9, ymmword ptr [rdx + 2240]
    vfmadd231ps ymm8, ymm9, ymmword ptr [rdx + 2272]
    vbroadcastss ymm9, dword ptr [rsi + 36]
    vfmadd231ps ymm1, ymm9, ymmword ptr [rdx + 2304]
    vfmadd231ps ymm2, ymm9, ymmword ptr [rdx + 2336]
    vfmadd231ps ymm3, ymm9, ymmword ptr [rdx + 2368]
    vfmadd231ps ymm4, ymm9, ymmword ptr [rdx + 2400]
    vfmadd231ps ymm5, ymm9, ymmword ptr [rdx + 2432]
    vfmadd231ps ymm6, ymm9, ymmword ptr [rdx + 2464]
    vfmadd231ps ymm7, ymm9, ymmword ptr [rdx + 2496]
    vfmadd231ps ymm8, ymm9, ymmword ptr [rdx + 2528]
    vbroadcastss ymm9, dword ptr [rsi + 40]
    vfmadd231ps ymm1, ymm9, ymmword ptr [rdx + 2560]
    vfmadd231ps ymm2, ymm9, ymmword ptr [rdx + 2592]
    vfmadd231ps ymm3, ymm9, ymmword ptr [rdx + 2624]
    vfmadd231ps ymm4, ymm9, ymmword ptr [rdx + 2656]
    vfmadd231ps ymm5, ymm9, ymmword ptr [rdx + 2688]
    vfmadd231ps ymm6, ymm9, ymmword ptr [rdx + 2720]
    vfmadd231ps ymm7, ymm9, ymmword ptr [rdx + 2752]
    vfmadd231ps ymm8, ymm9, ymmword ptr [rdx + 2784]
    vbroadcastss ymm9, dword ptr [rsi + 44]
    vfmadd231ps ymm1, ymm9, ymmword ptr [rdx + 2816]
    vfmadd231ps ymm2, ymm9, ymmword ptr [rdx + 2848]
    vfmadd231ps ymm3, ymm9, ymmword ptr [rdx + 2880]
    vfmadd231ps ymm4, ymm9, ymmword ptr [rdx + 2912]
    vfmadd231ps ymm5, ymm9, ymmword ptr [rdx + 2944]
    vfmadd231ps ymm6, ymm9, ymmword ptr [rdx + 2976]
    vfmadd231ps ymm7, ymm9, ymmword ptr [rdx + 3008]
    vfmadd231ps ymm8, ymm9, ymmword ptr [rdx + 3040]
    vbroadcastss ymm9, dword ptr [rsi + 48]
    vfmadd231ps ymm1, ymm9, ymmword ptr [rdx + 3072]
    vfmadd231ps ymm2, ymm9, ymmword ptr [rdx + 3104]
    vfmadd231ps ymm3, ymm9, ymmword ptr [rdx + 3136]
    vfmadd231ps ymm4, ymm9, ymmword ptr [rdx + 3168]
    vfmadd231ps ymm5, ymm9, ymmword ptr [rdx + 3200]
    vfmadd231ps ymm6, ymm9, ymmword ptr [rdx + 3232]
    vfmadd231ps ymm7, ymm9, ymmword ptr [rdx + 3264]
    vfmadd231ps ymm8, ymm9, ymmword ptr [rdx + 3296]
    vbroadcastss ymm9, dword ptr [rsi + 52]
    vfmadd231ps ymm1, ymm9, ymmword ptr [rdx + 3328]
    vfmadd231ps ymm2, ymm9, ymmword ptr [rdx + 3360]
    vfmadd231ps ymm3, ymm9, ymmword ptr [rdx + 3392]
    vfmadd231ps ymm4, ymm9, ymmword ptr [rdx + 3424]
    vfmadd231ps ymm5, ymm9, ymmword ptr [rdx + 3456]
    vfmadd231ps ymm6, ymm9, ymmword ptr [rdx + 3488]
    vfmadd231ps ymm7, ymm9, ymmword ptr [rdx + 3520]
    vfmadd231ps ymm8, ymm9, ymmword ptr [rdx + 3552]
    vbroadcastss ymm9, dword ptr [rsi + 56]
    vfmadd231ps ymm1, ymm9, ymmword ptr [rdx + 3584]
    vfmadd231ps ymm2, ymm9, ymmword ptr [rdx + 3616]
    vfmadd231ps ymm3, ymm9, ymmword ptr [rdx + 3648]
    vfmadd231ps ymm4, ymm9, ymmword ptr [rdx + 3680]
    vfmadd231ps ymm5, ymm9, ymmword ptr [rdx + 3712]
    vfmadd231ps ymm6, ymm9, ymmword ptr [rdx + 3744]
    vfmadd231ps ymm7, ymm9, ymmword ptr [rdx + 3776]
    vfmadd231ps ymm8, ymm9, ymmword ptr [rdx + 3808]
    vbroadcastss ymm9, dword ptr [rsi + 60]
    vfmadd231ps ymm1, ymm9, ymmword ptr [rdx + 3840]
    vfmadd231ps ymm2, ymm9, ymmword ptr [rdx + 3872]
    vfmadd231ps ymm3, ymm9, ymmword ptr [rdx + 3904]
    vfmadd231ps ymm4, ymm9, ymmword ptr [rdx + 3936]
    vfmadd231ps ymm5, ymm9, ymmword ptr [rdx + 3968]
    vfmadd231ps ymm6, ymm9, ymmword ptr [rdx + 4000]
    vfmadd231ps ymm7, ymm9, ymmword ptr [rdx + 4032]
    vfmadd231ps ymm8, ymm9, ymmword ptr [rdx + 4064]
    vmaxps ymm1, ymm1, ymm0
    vmovups ymmword ptr [rdi], ymm1
    vmaxps ymm2, ymm2, ymm0
    vmovups ymmword ptr [rdi + 32], ymm2
    vmaxps ymm3, ymm3, ymm0
    vmovups ymmword ptr [rdi + 64], ymm3
    vmaxps ymm4, ymm4, ymm0
    vmovups ymmword ptr [rdi + 96], ymm4
    vmaxps ymm5, ymm5, ymm0
    vmovups ymmword ptr [rdi + 128], ymm5
    vmaxps ymm6, ymm6, ymm0
    vmovups ymmword ptr [rdi + 160], ymm6
    vmaxps ymm7, ymm7, ymm0
    vmovups ymmword ptr [rdi + 192], ymm7
    vmaxps ymm8, ymm8, ymm0
    vmovups ymmword ptr [rdi + 224], ymm8
    add rsi, 64
    add rdi, 256
    dec rcx
    jnz 2b
3:
    vzeroupper
    ret
