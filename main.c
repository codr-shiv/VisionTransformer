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
    print_tensor4(Images);
    Tensor4 ResizedImages = Resize256(Images);
    print_tensor4(ResizedImages);
    Tensor4 CroppedImages = Crop224(ResizedImages);
    print_tensor4(CroppedImages);
    Normalize(CroppedImages);
    print_tensor4(CroppedImages);

    // conv2d
    // tempdata
    Tensor4 patch_embed_proj_weights = alloc_tensor4(192, 3, 16, 16);
    Tensor1 patch_embed_proj_biases = alloc_tensor1(192);
    for (int i = 0; i < 192*3*16*16; i++) {
        patch_embed_proj_weights.data[i] = 1.0f;
    }
    for (int i = 0; i < 192; i++) {
        patch_embed_proj_biases.data[i] = 1.0f;
    }
    Tensor3 conv = Conv2D(CroppedImages, patch_embed_proj_weights, patch_embed_proj_biases);
    printf("[%d %d %d]\n", conv.B, conv.X, conv.D);
    print_tensor3(conv);
    
    free_tensor4(Images);
    free_tensor4(ResizedImages);
    free_tensor4(CroppedImages);
    free_tensor4(patch_embed_proj_weights);
    free_tensor1(patch_embed_proj_biases);
    free_tensor3(conv);
    free(Labels);
}