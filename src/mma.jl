# mma.sync wrappers

using Base: llvmcall

# TODO: Clean this up.
mma884_row_row(a, b, c) = llvmcall("""
                 %ah0i = insertelement <2 x half> undef, half %0, i32 0
                 %ah0f = insertelement <2 x half> %ah0i, half %1, i32 1
                 %ah1i = insertelement <2 x half> undef, half %2, i32 0
                 %ah1f = insertelement <2 x half> %ah1i, half %3, i32 1

                 %bh0i = insertelement <2 x half> undef, half %4, i32 0
                 %bh0f = insertelement <2 x half> %bh0i, half %5, i32 1
                 %bh1i = insertelement <2 x half> undef, half %6, i32 0
                 %bh1f = insertelement <2 x half> %bh1i, half %7, i32 1

                 %a0 = bitcast <2 x half> %ah0f to i32
                 %a1 = bitcast <2 x half> %ah1f to i32

                 %b0 = bitcast <2 x half> %bh0f to i32
                 %b1 = bitcast <2 x half> %bh1f to i32

                 %mma = call { float, float, float, float, float, float, float, float } asm sideeffect "mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 {\$0,\$1,\$2,\$3,\$4,\$5,\$6,\$7}, {\$8, \$9}, {\$10,\$11}, {\$12,\$13,\$14,\$15,\$16,\$17,\$18,\$19};", "=f,=f,=f,=f,=f,=f,=f,=f,r,r,r,r,f,f,f,f,f,f,f,f"(i32 %a0, i32 %a1, i32 %b0, i32 %b1, float %8, float %9, float %10, float %11, float %12, float %13, float %14, float %15)

                 %mma0 = extractvalue { float, float, float, float, float, float, float, float } %mma, 0
                 %mma1 = extractvalue { float, float, float, float, float, float, float, float } %mma, 1
                 %mma2 = extractvalue { float, float, float, float, float, float, float, float } %mma, 2
                 %mma3 = extractvalue { float, float, float, float, float, float, float, float } %mma, 3
                 %mma4 = extractvalue { float, float, float, float, float, float, float, float } %mma, 4
                 %mma5 = extractvalue { float, float, float, float, float, float, float, float } %mma, 5
                 %mma6 = extractvalue { float, float, float, float, float, float, float, float } %mma, 6
                 %mma7 = extractvalue { float, float, float, float, float, float, float, float } %mma, 7

                 %rv0 = insertvalue [8 x float] undef, float %mma0, 0
                 %rv1 = insertvalue [8 x float] %rv0, float %mma1, 1
                 %rv2 = insertvalue [8 x float] %rv1, float %mma2, 2
                 %rv3 = insertvalue [8 x float] %rv2, float %mma3, 3
                 %rv4 = insertvalue [8 x float] %rv3, float %mma4, 4
                 %rv5 = insertvalue [8 x float] %rv4, float %mma5, 5
                 %rv6 = insertvalue [8 x float] %rv5, float %mma6, 6
                 %rv7 = insertvalue [8 x float] %rv6, float %mma7, 7

                 ret [8 x float] %rv7
                 """,
                 NTuple{8, Float32},
                 Tuple{
                    Float16, Float16, Float16, Float16,
                    Float16, Float16, Float16, Float16,
                    Float32, Float32, Float32, Float32, Float32, Float32, Float32, Float32
                 },
                 Float16.(a)...,
                 Float16.(b)...,
                 Float32.(c)...)
