#pragma once
#include "vec3.cuh"

struct Ray {
    Vec3 origin;
    Vec3 dir;
};

__host__ __device__ Ray createRay(Vec3 origin, Vec3 dir) {
    Ray r; 
    
    r.origin = origin; 
    r.dir = dir;

    return r;
}

// where is this ray at distance t?
__host__ __device__ Vec3 rayAt(Ray r, float t) {
    return add(r.origin, scale(r.dir, t));
}