#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

float* loadImages(const char* path, int* count, int* size) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("Cannot open %s\n", path); exit(1); }

    int magic, rows, cols;
    fread(&magic,  4, 1, f); magic  = reverseInt(magic);
    fread(count,   4, 1, f); *count = reverseInt(*count);
    fread(&rows,   4, 1, f); rows   = reverseInt(rows);
    fread(&cols,   4, 1, f); cols   = reverseInt(cols);

    *size = rows * cols;
    float* data = new float[(*count) * (*size)];

    for (int i = 0; i < (*count) * (*size); i++) {
        unsigned char pixel;
        fread(&pixel, 1, 1, f);
        data[i] = pixel / 255.0f;
    }

    fclose(f);
    printf("Loaded %d images (%dx%d)\n", *count, rows, cols);
    return data;
}

float* loadLabels(const char* path, int* count) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("Cannot open %s\n", path); exit(1); }

    int magic;
    fread(&magic, 4, 1, f);
    fread(count,  4, 1, f); *count = reverseInt(*count);

    float* labels = new float[(*count) * 10]();

    for (int i = 0; i < *count; i++) {
        unsigned char label;
        fread(&label, 1, 1, f);
        labels[i * 10 + label] = 1.0f;
    }

    fclose(f);
    printf("Loaded %d labels\n", *count);
    return labels;
}