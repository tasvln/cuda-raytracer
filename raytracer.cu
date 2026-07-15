#include <cstdio>
#include <cstdlib>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "vec3.cuh"
#include "ray.cuh"
#include "sphere.cuh"

const int WIDTH = 512;
const int HEIGHT = 512;
const int NUM_SPHERES = 3;

__global__ void renderKernel(unsigned char* image, Sphere* spheres) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    int idx = (y * WIDTH + x) * 3;

    float u = (float(x) / WIDTH)  * 2.0f - 1.0f;
    float v = (float(y) / HEIGHT) * 2.0f - 1.0f;

    Ray ray = createRay(createVec3(0,0,0), normalize(createVec3(u, v, -1.0f)));

    // find closest sphere hit
    float closestT = 1e20f;
    int hitIdx = -1;
    for (int i = 0; i < NUM_SPHERES; i++) {
        float t = hitSphere(spheres[i], ray);
        if (t > 0.0f && t < closestT) {
            closestT = t;
            hitIdx = i;
        }
    }

    if (hitIdx >= 0) {
        Vec3 hitPoint = rayAt(ray, closestT);
        
        Vec3 normal = normalize(createVec3(
            hitPoint.x - spheres[hitIdx].center.x,
            hitPoint.y - spheres[hitIdx].center.y,
            hitPoint.z - spheres[hitIdx].center.z
        ));

        // Vec3 lightDir = normalize(createVec3(1.0f, 1.0f, 0.5f));

        // float brightness = dot(normal, lightDir);
        // if (brightness < 0.0f) brightness = 0.0f;
        //     brightness = 0.1f + 0.9f * brightness;

        Vec3 light1 = normalize(createVec3(1.0f, 1.0f, 0.5f));
        Vec3 light2 = normalize(createVec3(-1.0f, 0.5f, 0.3f));

        float b1 = dot(normal, light1);
        float b2 = dot(normal, light2);
        if (b1 < 0.0f) b1 = 0.0f;
        if (b2 < 0.0f) b2 = 0.0f;

        float brightness = 0.1f + 0.6f * b1 + 0.3f * b2;
        if (brightness > 1.0f) brightness = 1.0f;

        image[idx + 0] = (unsigned char)(spheres[hitIdx].color[0] * brightness);
        image[idx + 1] = (unsigned char)(spheres[hitIdx].color[1] * brightness);
        image[idx + 2] = (unsigned char)(spheres[hitIdx].color[2] * brightness);
    } else {
        float t2 = 0.5f * (ray.dir.y + 1.0f);
        image[idx + 0] = (unsigned char)((1.0f - t2) * 255 + t2 * 128);
        image[idx + 1] = (unsigned char)((1.0f - t2) * 255 + t2 * 178);
        image[idx + 2] = (unsigned char)((1.0f - t2) * 255 + t2 * 255);
    }
}

int main() {
    // define spheres on CPU first
    Sphere h_spheres[NUM_SPHERES];

    h_spheres[0] = createSphere(createVec3( 0.0f,  0.0f, -2.0f), 0.5f, 255, 50,  50);
    h_spheres[1] = createSphere(createVec3(-1.1f,  0.0f, -2.5f), 0.5f, 50,  255, 50);
    h_spheres[2] = createSphere(createVec3( 1.1f, -0.2f, -2.2f), 0.5f, 50,  50,  255);

    // copy spheres to GPU
    Sphere* d_spheres;
    cudaMalloc(&d_spheres, NUM_SPHERES * sizeof(Sphere));
    cudaMemcpy(d_spheres, h_spheres, NUM_SPHERES * sizeof(Sphere), cudaMemcpyHostToDevice);

    // allocate image buffer on GPU
    size_t bufferSize = WIDTH * HEIGHT * 3 * sizeof(unsigned char);
    unsigned char* d_image;
    cudaMalloc(&d_image, bufferSize);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        (WIDTH  + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    renderKernel<<<numBlocks, threadsPerBlock>>>(d_image, d_spheres);
    cudaDeviceSynchronize();

    unsigned char* h_image = (unsigned char*)malloc(bufferSize);
    cudaMemcpy(h_image, d_image, bufferSize, cudaMemcpyDeviceToHost);

    stbi_write_png("output.png", WIDTH, HEIGHT, 3, h_image, WIDTH * 3);
    printf("Saved output.png\n");

    cudaFree(d_image);
    cudaFree(d_spheres);
    free(h_image);
    return 0;
}