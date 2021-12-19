//
// Created by schrodinger on 11/2/21.
//
#pragma once

#include <random>
#include <utility>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>

static constexpr double COLLISION_RATIO = 0.01;

__device__
double delta_xx(double *x, int i, int j) {
    return x[i] - x[j];
}

__device__
double delta_yy(double *y, int i, int j) {
    return y[i] - y[j];
}

__device__
double distance_squaree(double *x, double *y, int i, int j) {
    auto delta_x = x[i] - x[j];
    auto delta_y = y[i] - y[j];
    return delta_x * delta_x + delta_y * delta_y;
}

__device__
double distancee(double *x, double *y, int i, int j) {
    return std::sqrt(distance_squaree(x, y, i, j));
}

__device__
bool collide(double *x, double *y, int i, int j, double radius) {
    return distance_squaree(x, y, i, j) <= radius * radius;
}

__device__
void check_and_update(
                    double *x, double *y,
                    double *vx,double *vy,
                    double *ax,double *ay,
                    double *m,
                    int i, int j, double radius, double gravity) {
    auto delta_x = delta_xx(x, i, j);
    auto delta_y = delta_yy(y, i, j);
    auto distance_square = distance_squaree(x,y,i,j);
    auto ratio = 1 + COLLISION_RATIO;
    if (distance_square < radius * radius) {
        distance_square = radius * radius;
    }
    auto distance = distancee(x,y,i,j);
    if (distance < radius) {
        distance = radius;
    }
    if (collide(x,y,i,j, radius)) {
        auto dot_prod = delta_x * (vx[i] - vx[j])
                        + delta_y * (vy[i] - vy[j]);
        auto scalar = 2 / (m[i] + m[j]) * dot_prod / distance_square;
        vx[i] -= scalar * delta_x * m[j];
        vy[i] -= scalar * delta_y * m[j];
        vx[j] += scalar * delta_x * m[i];
        vy[j] += scalar * delta_y * m[i];
        // now relax the distance a bit: after the collision, there must be
        // at least (ratio * radius) between them
        x[i] += delta_x / distance * ratio * radius / 2.0;
        y[i] += delta_y / distance * ratio * radius / 2.0;
        x[j] -= delta_x / distance * ratio * radius / 2.0;
        y[j] -= delta_y / distance * ratio * radius / 2.0;
    } else {
        // update acceleration only when no collision
        auto scalar = gravity / distance_square / distance;
        ax[i] -= scalar * delta_x * m[j];
        ay[i] -= scalar * delta_y * m[j];
        ax[j] += scalar * delta_x * m[i];
        ay[j] += scalar * delta_y * m[i];
    }
}

__device__
void handle_wall_collision(
                            double *x, double *y,
                            double *vx,double *vy,
                            double *ax,double *ay,
                            double *m, int i,
                            double position_range, double radius) {
    bool flag = false;
    if (x[i] <= radius) {
        flag = true;
        x[i] = radius + radius * COLLISION_RATIO;
        vx[i] = -vx[i];
    } else if (x[i] >= position_range - radius) {
        flag = true;
        x[i] = position_range - radius - radius * COLLISION_RATIO;
        vx[i] = -vx[i];
    }

    if (y[i] <= radius) {
        flag = true;
        y[i] = radius + radius * COLLISION_RATIO;
        vy[i] = -vy[i];
    } else if (y[i] >= position_range - radius) {
        flag = true;
        y[i] = position_range - radius - radius * COLLISION_RATIO;
        vy[i] = -vy[i];
    }
    if (flag) {
        ax[i] = 0;
        ay[i] = 0;
    }
}

__device__
void update_for_tick_write(
        double *x, double *y,
        double *vx,double *vy,
        double *ax,double *ay,
        double *m, int i,
        double elapse,
        double position_range,
        double radius) {
    vx[i] += ax[i] * elapse;
    vy[i] += ay[i] * elapse;
    handle_wall_collision(x,y,vx,vy,ax,ay,m,i,position_range, radius);
    x[i] += vx[i] * elapse;
    y[i] += vy[i] * elapse;
    handle_wall_collision(x,y,vx,vy,ax,ay,m,i,position_range, radius);
}


__device__
int getBlockId() {
  return blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
}

__device__
int getLocalThreadId() {
  return (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
}

__device__
int getThreadId() {
  int blockId = getBlockId();
  int localThreadId = getLocalThreadId();
  return blockId * (blockDim.x * blockDim.y * blockDim.z) + localThreadId;
}

__global__
void cuda_setnum() {
  if (getLocalThreadId() != 0) return;
  
}

__global__
void update_for_tick(double *x, double *y,
                    double *vx,double *vy,
                    double *ax,double *ay,
                    double *m,
                    double elapse,
                    double gravity,
                    double position_range,
                    double radius,
                    int local_bodies,
                    int bodies) {
    int thread_id = getThreadId();
    // printf("Hi. My threadId=%d\n", thread_id);
    for(int i=(thread_id)*local_bodies; i<min(bodies, (thread_id+1)*local_bodies); i++){
        ax[i] = 0;
        ay[i] = 0;
    }
    for (int i = (thread_id)*local_bodies; i < (thread_id+1)*local_bodies; ++i) {
        if(i<bodies){
            for (int j = i + 1; j < bodies; ++j) {
                if(j<bodies){
                    check_and_update(x,y,vx,vy,ax,ay,m,i, j, radius, gravity);
                }
            }
        }
    }
    for (int i=(thread_id)*local_bodies; i < (thread_id+1)*local_bodies; ++i) {
        if(i<bodies){
            update_for_tick_write(x,y,vx,vy,ax,ay,m,i, elapse, position_range, radius);
        }
    }
}

