#include <cstdio>
#include <cstdlib>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

const int WIDTH = 512;
const int HEIGHT = 512;

// Vector 3 (x,y,z)
struct Vec3 {
    float x, y, z;
};

// __device__ means this function runs on the GPU
__device__ Vec3 createVec3(float x, float y, float z) {
    Vec3 v; v.x = x; v.y = y; v.z = z;
    return v;
}

// bound length with sqrtf(x²+y²+z²) 
// Dividing each component by that length rescales the vector to length 1 while keeping its direction the same.
__device__ Vec3 normalize(Vec3 v) {
    float len = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    return createVec3(v.x/len, v.y/len, v.z/len);
}

__global__ void renderKernel(unsigned char* image) {
    // same calculation to get the global unique index of a thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT) return;

    // uv mapping like textures -> just coordinates
    float u = (float(x) / WIDTH) * 2.0f - 1.0f;
    float v = (float(y) / HEIGHT) * 2.0f - 1.0f;

    // camera sits at the world origin, (0,0,0)
    Vec3 rayOrigin = createVec3(0.0f, 0.0f, 0.0f);

    // rayDir — x = u, y = v, z = -1. The -1 is "one unit forward into the screen" 
    // (negative z = forward, by convention). u and v tilt that direction left/right and 
    // up/down depending on which pixel we're on. Center pixel (u=0,v=0) 
    // points straight at (0,0,-1) — dead ahead. Edge pixels tilt outward, 
    // which is exactly what creates the camera's field of view / perspective.
    Vec3 rayDir = createVec3(u, v, -1.0f);
    rayDir = normalize(rayDir);

    // Each color channel lerps (linearly interpolates) between white (255,255,255) 
    // at the bottom (t=0) and a soft blue (128,178,255) at the top (t=1). 
    // The formula (1-t)*A + t*B is the standard lerp pattern.
    
    float t = 0.5f * (rayDir.y + 1.0f);
    unsigned char r = (unsigned char)((1.0f - t) * 255 + t * 128);
    unsigned char g = (unsigned char)((1.0f - t) * 255 + t * 178);
    unsigned char b = (unsigned char)((1.0f - t) * 255 + t * 255);

    // 3 bytes per pixel (RGB)
    int idx = (y * WIDTH + x) * 3;

    image[idx + 0] = r;
    image[idx + 1] = g;
    image[idx + 2] = b;
}

int main() {
    int numPixels = WIDTH * HEIGHT;
    size_t bufferSize = numPixels * 3 * sizeof(unsigned char);

    // create image on cpu and gpu
    unsigned char* d_image;
    cudaMalloc(&d_image, bufferSize);

    // dim3 is just a 3D version of a number (x, y, z) — used here for 2D since we're dealing with an image, not a 1D list
    // essencially vec3?
    
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        (WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    renderKernel<<<numBlocks, threadsPerBlock>>>(d_image);
    cudaDeviceSynchronize();

    // h_ = host pointer
    unsigned char* h_image = (unsigned char*)malloc(bufferSize);
    cudaMemcpy(h_image, d_image, bufferSize, cudaMemcpyDeviceToHost);

    stbi_write_png("output.png", WIDTH, HEIGHT, 3, h_image, WIDTH * 3);
    printf("Saved output.png\n");

    cudaFree(d_image);
    free(h_image);

    return 0;
}