#define PROFILE 1
#include "prelude.h"
#include "profiler/profiler.h"

constexpr u32 TestingScalarSize = 4;
using TestingFloat = fp8::E4M3Type;

template <typename Dimensions>
using TestingBlock = Block<TestingScalarSize, Dimensions, TestingFloat,
                           CPUArithmetic, SharedExponentQuantization>;



// ----------------------------------------                       
//          UTILITY FUNCTIONS
// ----------------------------------------

// block_to_array_3d : convert a 3Dim blockedFP format into a 3D array of f64
template <size_t N>
static std::array<std::array<std::array<f64, N>, N>, N> block_to_array_3d(
    const TestingBlock<BlockDims<N,N,N>> block){
        using Dimensions = BlockDims<N,N,N>;
        std::array<std::array<std::array<f64, N>, N>, N> out{};
        for (u32 i = 0; i < N; i++) {
            for (u32 j = 0; j < N; j++) {
                for (u32 k = 0; k < N; k++) {
                    const u32 linear = Dimensions::CoordsToLinear({i, j, k});
                    out[i][j][k] = block.RealizeAtUnsafe(linear); 
                }
            }
        }
        return out;
    }


// max_error_3d : compute maximum absolute difference between two 3D arrays of the same size
template <size_t N>
f64 max_abs_error_3d(
    const std::array<std::array<std::array<f64,N>,N>,N> A,
    const std::array<std::array<std::array<f64,N>,N>,N> B){
        f64 maxErr = 0.0;
        for (size_t i = 0; i < N; i++){
            for (size_t j = 0; j < N; j++){
                for (size_t k = 0; k < N; k++){
                    maxErr = std::max(maxErr, std::abs(A[i][j][k] - B[i][j][k]));
                }
            }
        }          
        return maxErr;
    }

// mean_abs_error_3d : compute the avg element-wise absolute diff between 3D arrays of the same size
template <size_t N>
f64 mean_abs_error_3d(
    const std::array<std::array<std::array<f64,N>,N>,N> A,
    const std::array<std::array<std::array<f64,N>,N>,N> B){
    f64 sumAbs = 0.0;
    size_t count = 0;
    
    for (size_t i = 0; i < N; i++){
        for (size_t j = 0; j < N; j++){
            for (size_t k = 0; k < N; k++){
                sumAbs += std::abs(A[i][j][k] - B[i][j][k]);
                count++;
            }
        }
    }          
    return sumAbs / count;
}

// ----------------------------------------   
//    HEAT 3D STENCIL POLYBENCH VERSION
// ----------------------------------------   
// ref: https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/stencils/heat-3d/heat-3d.c
template <size_t N>
static std::array<std::array<std::array<f64, N>, N>, N> heat_3d_poly(
    const int steps,
    std::array<std::array<std::array<f64, N>, N>, N> A,
    std::array<std::array<std::array<f64, N>, N>, N> B){
        profiler::func();
        #pragma scop
            for (int t = 1; t <= steps; t++) {
                for (int i = 1; i < N-1; i++) {
                    for (int j = 1; j < N-1; j++) {
                        for (int k = 1; k < N-1; k++) {
                            B[i][j][k] =  0.125 * (A[i+1][j][k] - 2.0 * A[i][j][k] + A[i-1][j][k])
                                        + 0.125 * (A[i][j+1][k] - 2.0 * A[i][j][k] + A[i][j-1][k])
                                        + 0.125 * (A[i][j][k+1] - 2.0 * A[i][j][k] + A[i][j][k-1])
                                        + A[i][j][k];
                        }
                    }
                }
                for (int i = 1; i < N-1; i++) {
                    for (int j = 1; j < N-1; j++) {
                        for (int k = 1; k < N-1; k++) {
                            A[i][j][k] =   0.125 * (B[i+1][j][k] - 2.0 * B[i][j][k] + B[i-1][j][k])
                                            + 0.125 * (B[i][j+1][k] - 2.0 * B[i][j][k] + B[i][j-1][k])
                                            + 0.125 * (B[i][j][k+1] - 2.0 * B[i][j][k] + B[i][j][k-1])
                                            + B[i][j][k];
                        }
                    }
                }
            }
        #pragma endscop
        return A;
        }


// ----------------------------------------   
//    HEAT 3D STENCIL BLOCK VERSION
// ----------------------------------------  
template <size_t N>
static TestingBlock<BlockDims<N,N,N>> heat_3d_block(
    const int steps,
    TestingBlock<BlockDims<N,N,N>> A, 
    TestingBlock<BlockDims<N,N,N>> B){
        profiler::func();
        using Dimensions = BlockDims<N,N,N>;
        
        for (u32 t=0; t<steps;t++){
            for (u32 i=1;i<N-1;i++){
                for (u32 j=1;j<N-1;j++){
                    for (u32 k=1;k<N-1;k++){
                        u32 idx = Dimensions::CoordsToLinear({i,j,k});
                        f64 newVal = 0.125*(A[i+1,j,k] - 2.0*A[i,j,k] + A[i-1,j,k])
                                + 0.125*(A[i,j+1,k] - 2.0*A[i,j,k] + A[i,j-1,k])
                                + 0.125*(A[i,j,k+1] - 2.0*A[i,j,k] + A[i,j,k-1])
                                + A[i,j,k];
                        B.SetValue(idx, newVal);
                    }
                }
            }

            for (u32 i=1;i<N-1;i++){
                for (u32 j=1;j<N-1;j++){
                    for (u32 k=1;k<N-1;k++){
                        u32 idx = Dimensions::CoordsToLinear({i,j,k});
                        f64 newVal = 0.125*(B[i+1,j,k] - 2.0*B[i,j,k] + B[i-1,j,k])
                                + 0.125*(B[i,j+1,k] - 2.0*B[i,j,k] + B[i,j-1,k])
                                + 0.125*(B[i,j,k+1] - 2.0*B[i,j,k] + B[i,j,k-1])
                                + B[i,j,k];
                        A.SetValue(idx, newVal);
                    }
                }
            }
    }
    return A;
}



constexpr u32 N = 32; //grid size (32x32x32)

int main(){

    int tsteps = 20; 
    profiler::begin();

    using Size = BlockDims<N,N,N>;
    using Block = TestingBlock<Size>;


    //init f64 arrays
    std::array<std::array<std::array<f64, N>, N>, N> a{};
    std::array<std::array<std::array<f64, N>, N>, N> b{};
    for (u32 i = 0; i < N; i++){
            for (u32 j = 0; j < N; j++){
                for (u32 k = 0; k < N; k++){
                    a[i][j][k] = (i + j + (N-k)) * 10.0 / N;
                    b[i][j][k] = (i + j + (N-k)) * 10.0 / N;
                }
            }
        }
                    
    // init linearized arrays for blocks
    std::array<f64,N*N*N> aLinear{};
    std::array<f64,N*N*N> bLinear{};
    for (u32 i = 0; i < N; i++) {
        for (u32 j = 0; j < N; j++) {
            for (u32 k = 0; k<N;k++){
                aLinear[i*N*N + j*N + k] = a[i][j][k]; // aLinear[..] = a[..]
                bLinear[i*N*N + j*N + k] = b[i][j][k];
            }
        }
    }

    const auto blockA = Block(aLinear);
    const auto blockB = Block(bLinear);


    auto true_result = heat_3d_poly<N>(tsteps, a, b);           // run heat3d polybench
    auto blockResult = heat_3d_block<N>(tsteps,blockA, blockB); // run heat3d block version

    profiler::end_and_print();

    //get elapsed seconds for both
    double poly_time = profiler::get_elapsed_seconds("heat_3d_poly");
    double block_time = profiler::get_elapsed_seconds("heat_3d_block");

    std::cout << "Poly time: " << poly_time << " seconds\n";
    std::cout << "Block time: " << block_time << " seconds\n";

    //convert blockresult to array so that we can calculate the error
    auto blockResultToarray = block_to_array_3d<N>(blockResult);

    //calculate error(s)
    f64 mean_err = mean_abs_error_3d<N>(true_result, blockResultToarray);
    f64 max_err = max_abs_error_3d<N>(true_result, blockResultToarray);

    std::cout << "Mean absolute error: " << mean_err << "\n";
    std::cout << "Max absolute error: " << max_err << "\n";
    
    return 0;
}