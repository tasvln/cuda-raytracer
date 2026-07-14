#pragma once

struct Vec3 {
    float x, y, z;
};

__host__ __device__ Vec3 createVec3(float x, float y, float z) {
    Vec3 v; 

    v.x = x; 
    v.y = y; 
    v.z = z;

    return v;
}

__host__ __device__ Vec3 normalize(Vec3 v) {
    float len = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);

    return createVec3(v.x/len, v.y/len, v.z/len);
}

__host__ __device__ float dot(Vec3 a, Vec3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__host__ __device__ Vec3 add(Vec3 a, Vec3 b) {
    return createVec3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__host__ __device__ Vec3 scale(Vec3 v, float t) {
    return createVec3(v.x*t, v.y*t, v.z*t);
}