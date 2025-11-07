#include "../include/funcs.h"
#include <math.h>
#include <stdio.h>

Tensor4 split_heads(Tensor3 in, int H) {
    int B = in.B, X = in.X, D = in.D;
    int Y = D / H;

    Tensor4 out = alloc_tensor4(B, H, X, Y);

    for(int b=0; b<B; b++)
        for(int x=0; x<X; x++)
            for(int h=0; h<H; h++)
                for(int y=0; y<Y; y++)
                    T4(out,b,h,x,y) = T3(in,b,x,h*Y + y);
    return out;
}

Tensor3 merge_heads(Tensor4 in) {
    int B = in.B, H = in.H, X = in.X, Y = in.Y;
    Tensor3 out = alloc_tensor3(B, X, H*Y);

    for(int b=0; b<B; b++)
        for(int x=0; x<X; x++)
            for(int h=0; h<H; h++)
                for(int y=0; y<Y; y++)
                    T3(out,b,x,h*Y + y) = T4(in,b,h,x,y);

    return out;
}

Tensor4 qk_dot(Tensor4 Q, Tensor4 K) {
    int B=Q.B, H=Q.H, X=Q.X, Y=Q.Y;
    Tensor4 Scores = alloc_tensor4(B,H,X,X);

    for(int b=0; b<B; b++)
        for(int h=0; h<H; h++)
            for(int i=0; i<X; i++)
                for(int j=0; j<X; j++){
                    float sum = 0.0f;
                    for(int d=0;d<Y;d++)
                        sum += T4(Q,b,h,i,d) * T4(K,b,h,j,d);
                    T4(Scores,b,h,i,j) = sum;
                }
    return Scores;
}

void scale_scores(Tensor4 Scores) {
    float s = 1.0f / sqrtf((float)Scores.Y);
    int B=Scores.B, H=Scores.H, X=Scores.X;

    for(int b=0; b<B; b++)
        for(int h=0; h<H; h++)
            for(int i=0; i<X; i++)
                for(int j=0; j<X; j++)
                    T4(Scores,b,h,i,j) *= s;
}

void softmax_scores(Tensor4 Scores) {
    int B=Scores.B, H=Scores.H, X=Scores.X;

    for(int b=0; b<B; b++)
        for(int h=0; h<H; h++)
            for(int i=0; i<X; i++){

                float maxv = -1e30f;
                for(int j=0; j<X; j++)
                    if(T4(Scores,b,h,i,j) > maxv) maxv = T4(Scores,b,h,i,j);

                float sum = 0.0f;
                for(int j=0; j<X; j++){
                    T4(Scores,b,h,i,j) = expf(T4(Scores,b,h,i,j) - maxv);
                    sum += T4(Scores,b,h,i,j);
                }

                for(int j=0; j<X; j++)
                    T4(Scores,b,h,i,j) /= sum;
            }
}



Tensor3 layernorm(Tensor3 in, Tensor1 gamma, Tensor1 beta) {
    int B = in.B, X = in.X, D = in.D;
    Tensor3 Out = alloc_tensor3(B, X, D);

    for (int b = 0; b < B; b++) {
        for (int x = 0; x < X; x++) {
            float mean = 0.0f;
            float M2 = 0.0f;
            for (int d = 0; d < D; d++) {
                float val = T3(in, b, x, d);
                float delta = val - mean;
                float count = (float)(d + 1);
                mean += delta / count;
                float delta2 = val - mean;
                M2 += delta * delta2;
            }

            float var = M2 / (float)D;
            float inv_std = 1.0f / sqrtf(var + EPS);

            for (int d = 0; d < D; d++) {
                float val = T3(in, b, x, d);
                float normed = (val - mean) * inv_std;
                T3(Out, b, x, d) = normed * T1(gamma, d) + T1(beta, d);
            }
        }
    }
    return Out;
}


Tensor4 av_dot(Tensor4 Scores, Tensor4 V) {
    int B=Scores.B, H=Scores.H, X=Scores.X, Y=V.Y;
    Tensor4 Out = alloc_tensor4(B,H,X,Y);

    for(int b=0; b<B; b++)
        for(int h=0; h<H; h++)
            for(int i=0; i<X; i++)
                for(int y=0; y<Y; y++){
                    float sum = 0.0f;
                    for(int j=0; j<X; j++)
                        sum += T4(Scores,b,h,i,j) * T4(V,b,h,j,y);
                    T4(Out,b,h,i,y) = sum;
                }
    return Out;
}

// Compute row-wise stable softmax on Matrix input, store in output
void softmax_matrix(Matrix input, Matrix output) {
    if (input.rows != output.rows || input.cols != output.cols) {
        fprintf(stderr, "Error: softmax_matrix() dimension mismatch\n");
        return;
    }

    for (int r = 0; r < input.rows; r++) {
        // Step 1: find max in this row
        float max_val = -INFINITY;
        for (int c = 0; c < input.cols; c++) {
            if (M(input, r, c) > max_val)
                max_val = M(input, r, c);
        }

        // Step 2: compute exponentials of (x - max)
        float sum_exp = 0.0f;
        for (int c = 0; c < input.cols; c++) {
            float e = expf(M(input, r, c) - max_val);
            M(output, r, c) = e;
            sum_exp += e;
        }

        // Step 3: normalize by sum of exps
        for (int c = 0; c < input.cols; c++) {
            M(output, r, c) /= sum_exp;
        }
    }
}



Matrix transpose_matrix(Matrix W) {
    Matrix T = alloc_matrix(W.cols, W.rows);
    for (int r = 0; r < W.rows; r++) 
        for (int c = 0; c < W.cols; c++)
            M(T, c, r) = M(W, r, c);
    return T;
}

Tensor3 gemm(Tensor3 A, Matrix W) {
    int B = A.B;
    int S = A.X;
    int D_in = A.D; 
    int D_out = W.cols;

    Tensor3 Out = alloc_tensor3(B, S, D_out);

    for (int b = 0; b < B; b++)
        for (int s = 0; s < S; s++)
            for (int d_out = 0; d_out < D_out; d_out++) {
                float sum = 0.0f;
                for (int d_in = 0; d_in < D_in; d_in++) 
                    sum += T3(A, b, s, d_in) * M(W, d_in, d_out);
                T3(Out, b, s, d_out) = sum;
            }
        
    return Out;
}

void add_bias(Tensor3 A, Tensor1 bias) {
    int B = A.B, X = A.X, D = A.D;
    for (int b = 0; b < B; b++) 
        for (int x = 0; x < X; x++)
            for (int d = 0; d < D; d++)
                T3(A, b, x, d) += T1(bias, d);
}

// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
void gelu(Tensor3 A) {
    size_t total_elements = (size_t)A.B * A.X * A.D;

    for (size_t i = 0; i < total_elements; i++) {
        float x = A.data[i];
        float cube = 0.044715f * x * x * x;
        float inner = 0.79788456f * (x + cube);
        A.data[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}


Tensor3 mlp_forward(Tensor3 in, MLP_Weights* weights) {

    Matrix fc1_wT = transpose_matrix(weights->fc1_weight);
    Matrix fc2_wT = transpose_matrix(weights->fc2_weight);
    
    // [1, 197, 192] * [192, 768] = [1, 197, 768]
    Tensor3 fc1_out = gemm(in, fc1_wT);
    add_bias(fc1_out, weights->fc1_bias); 
    gelu(fc1_out); 

    // [1, 197, 768] * [768, 192] = [1, 197, 192]
    Tensor3 fc2_out = gemm(fc1_out, fc2_wT);
    add_bias(fc2_out, weights->fc2_bias);
    
    free_matrix(fc1_wT);
    free_matrix(fc2_wT);
    free_tensor3(fc1_out);

    return fc2_out;
}