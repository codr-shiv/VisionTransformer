#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "include/ViT.h"

int main() {
    char *Labels = malloc(sizeof(char) * DATASET_SIZE);
    // (B, C, H, W);
    // Tensor4 Images = LoadCIFAR10Dataset("dataset/cifar-10-batches-bin/train_all.bin", Labels, 0);
    Tensor4 Images = LoadImageFromPPM("dataset/ImageNetSelected/n01687978_10071.ppm");
    printf("Input Image\n");
    print_tensor4(Images);
    Tensor4 ResizedImages = Resize256(Images);
    printf("\n\nResized Image\n");
    print_tensor4(ResizedImages);
    Tensor4 CroppedImages = Crop224(ResizedImages);
    printf("\n\nCropped Image\n");
    print_tensor4(CroppedImages);
    Normalize(CroppedImages);
    printf("\n\nNormalized Image\n");
    print_tensor4(CroppedImages);

    // conv2d
    // tempdata

    // Tensor4 patch_embed_proj_weights = alloc_tensor4(192, 3, 16, 16);
    // Tensor1 patch_embed_proj_biases = alloc_tensor1(192);
    // for (int i = 0; i < 192*3*16*16; i++) {
    //     patch_embed_proj_weights.data[i] = 1.0f;
    // }
    // for (int i = 0; i < 192; i++) {
    //     patch_embed_proj_biases.data[i] = 1.0f;
    // }

    Tensor4 patch_embed_proj_weights = GetData4("parameters/patch_embed_proj_weight.bin");
    Tensor1 patch_embed_proj_biases = GetData1("parameters/patch_embed_proj_bias.bin");

    Tensor3 conv = Conv2D(CroppedImages, patch_embed_proj_weights, patch_embed_proj_biases);
    printf("[%d %d %d]\n", conv.B, conv.X, conv.D);
    printf("\n\nConvoltioned Image\n");
    print_tensor3(conv);

    Tensor3 cls = GetData3("parameters/cls_token.bin");
    Tensor3 pos_embed = GetData3("parameters/pos_embed.bin");
    printf("\n\nPatch Embedded Image\n");
    Tensor3 PreprocessedInputs = addCLSToken(conv, cls, pos_embed);
    print_tensor3(PreprocessedInputs);
    

    // ========================= PUT INTO LOOPS LATER =========================
    for (int e = 0; e < ENCODER_BLOCKS; e++) {
        Tensor1 zero_norm_weight = GetData1(GetPath(e, "norm1_weight"));
        Tensor1 zero_norm_bias = GetData1(GetPath(e, "norm1_bias"));

        PreprocessedInputs = layernorm(PreprocessedInputs, zero_norm_weight, zero_norm_bias);
        printf("\n\nNorm1ed Image\n");
        print_tensor3(PreprocessedInputs);

        Matrix qkv_weight = GetData2(GetPath(e, "attn_qkv_weight"));
        Matrix qkv_bias = GetData2(GetPath(e, "attn_qkv_bias"));

        Tensor3 MHAOutput = MHA(PreprocessedInputs, qkv_weight, qkv_bias);
        print_tensor3(PreprocessedInputs);

        free_tensor1(zero_norm_weight);
        free_tensor1(zero_norm_bias);
        free_matrix(qkv_weight);
        free_matrix(qkv_bias);
        free_tensor3(MHAOutput);
    }
    // ========================= PUT INTO LOOPS LATER =========================
    
    free_tensor4(Images);
    free_tensor4(ResizedImages);
    free_tensor4(CroppedImages);
    free_tensor4(patch_embed_proj_weights);
    free_tensor1(patch_embed_proj_biases);
    free_tensor3(cls);
    free_tensor3(pos_embed);
    free_tensor3(conv);
    free_tensor3(PreprocessedInputs);
    free(Labels);
}