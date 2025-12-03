template <typename T>
struct GPUArithmeticNaive {
    // Only arithmetic operations are done on GPU, not Spread
    static T PointwiseOpNaive(const T& lhs, const T& rhs, auto op) {
        constexpr size_t N = T::Length();
        auto l = lhs.Spread();
        auto r = rhs.Spread();
        std::array<ElemType, N> result;

        ElemType *d_l, *d_r, *d_result;
        cudaMalloc(&d_l, N * sizeof(ElemType));
        cudaMalloc(&d_r, N * sizeof(ElemType));
        cudaMalloc(&d_result, N * sizeof(ElemType));

        cudaMemcpy(d_l, l.data(), N * sizeof(ElemType), cudaMemcpyHostToDevice);
        cudaMemcpy(d_r, r.data(), N * sizeof(ElemType), cudaMemcpyHostToDevice);

        LaunchArithmeticKernel(d_l, d_r, d_result, N, op);

        cudaMemcpy(result.data(), d_result, N * sizeof(ElemType), cudaMemcpyDeviceToHost);

        cudaFree(d_l);
        cudaFree(d_r);
        cudaFree(d_result);

        return T(result);
    }
    static T Add(const T& lhs, const T& rhs) { return PointwiseOpNaive(lhs, rhs, ArithmeticOp::Add); }
    static T Sub(const T& lhs, const T& rhs) { return PointwiseOpNaive(lhs, rhs, ArithmeticOp::Sub); }
    static T Mul(const T& lhs, const T& rhs) { return PointwiseOpNaive(lhs, rhs, ArithmeticOp::Mul); }
    static T Div(const T& lhs, const T& rhs) { return PointwiseOpNaive(lhs, rhs, ArithmeticOp::Div); }
};