#pragma once
#include "vec3.cuh"
#include "ray.cuh"

struct Sphere {
    Vec3 center;
    float radius;
    unsigned char color[3];
};

__host__ __device__ Sphere createSphere(
    Vec3 center, 
    float radius, 
    unsigned char r, 
    unsigned char g, 
    unsigned char b
) {
    Sphere s;
    s.center = center;
    s.radius = radius;
    s.color[0] = r;
    s.color[1] = g;
    s.color[2] = b;
    return s;
}

__host__ __device__ float hitSphere(Sphere s, Ray ray) {
    Vec3 oc = createVec3(
        ray.origin.x - s.center.x,
        ray.origin.y - s.center.y,
        ray.origin.z - s.center.z
    );

    float a = dot(ray.dir, ray.dir);
    float b = 2.0f * dot(oc, ray.dir);
    float c = dot(oc, oc) - s.radius * s.radius;

    float discriminant = b*b - 4*a*c;
    if (discriminant < 0) return -1.0f;
    return (-b - sqrtf(discriminant)) / (2.0f * a);
}