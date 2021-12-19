#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <nbody/body.hpp>
#include <iostream>
#include <random>
#include <utility>
#include <time.h>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>

using namespace std;

template <typename ...Args>
void UNUSED(Args&&... args [[maybe_unused]]) {}

// dim3 grid_size;
// grid_size.x = 1;
// grid_size.y = 1;
// grid_size.z = 1;

// dim3 block_size;
// block_size.x = 1;
// block_size.y = 1;
// block_size.z = 1;

// int thread_nums = grid_size.x*grid_size.y*grid_size.z*block_size.x*block_size.y*block_size.z;

void initArray(double *x, double *y, double *m, int size, float position_range, float mass_range){
    srand((unsigned)time(NULL)); 
    for(int i=0; i<size; i++){
        x[i] = (rand() % int(position_range)) + 1;
    }
    for(int i=0; i<size; i++){
        y[i] = (rand() % int(position_range)) + 1;
    }
    for(int i=0; i<size; i++){
        m[i] = (rand() % int(mass_range)) + 1;
    }
}

int main(int argc, char **argv) {
    // UNUSED(argc, argv);
    int thread_num;
    if (argc < 2){
        // if user did not provide the size of array
        // the defualt value is set to be 100
        thread_num = 100;
    }else{
        thread_num = atoi(argv[1]);
    }
    static float gravity = 100;
    static float space = 800;
    static float radius = 5;
    static int bodies = 500;
    static float elapse = 0.001;
    static ImVec4 color = ImVec4(1.0f, 1.0f, 0.4f, 1.0f);
    static float max_mass = 50;
    static float current_space = space;
    static float current_max_mass = max_mass;
    static int current_bodies = bodies;
    int count = 0;
    double count_runtime = 0;
    // BodyPool pool(static_cast<size_t>(bodies), space, max_mass);
    double *hostX = new double[bodies];
    double *hostY = new double[bodies];
    double *hostVX = new double[bodies];
    double *hostVY = new double[bodies];
    double *hostAX = new double[bodies];
    double *hostAY = new double[bodies];
    double *hostM = new double[bodies];
    double *x;
    cudaMalloc(&x, sizeof(double) * bodies);
    double *y;
    cudaMalloc(&y, sizeof(double) * bodies);
    double *vx;
    cudaMalloc(&vx, sizeof(double) * bodies);
    double *vy;
    cudaMalloc(&vy, sizeof(double) * bodies);
    double *ax;
    cudaMalloc(&ax, sizeof(double) * bodies);
    double *ay;
    cudaMalloc(&ay, sizeof(double) * bodies);
    double *m;
    cudaMalloc(&m, sizeof(double) * bodies);
    initArray(hostX, hostY, hostM, bodies ,space, max_mass);
    cudaMemcpy(x, hostX, sizeof(double)*bodies, cudaMemcpyHostToDevice);
    cudaMemcpy(y, hostY, sizeof(double)*bodies, cudaMemcpyHostToDevice);
    cudaMemcpy(vx, hostVX, sizeof(double)*bodies, cudaMemcpyHostToDevice);
    cudaMemcpy(vy, hostVY, sizeof(double)*bodies, cudaMemcpyHostToDevice);
    cudaMemcpy(ax, hostAX, sizeof(double)*bodies, cudaMemcpyHostToDevice);
    cudaMemcpy(ay, hostAY, sizeof(double)*bodies, cudaMemcpyHostToDevice);
    cudaMemcpy(m, hostM, sizeof(double)*bodies, cudaMemcpyHostToDevice);
    graphic::GraphicContext context{"Assignment 2"};
    context.run([&](graphic::GraphicContext *context [[maybe_unused]], SDL_Window *) {
        auto io = ImGui::GetIO();
        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::Begin("Assignment 2", nullptr,
                     ImGuiWindowFlags_NoMove
                     | ImGuiWindowFlags_NoCollapse
                     | ImGuiWindowFlags_NoTitleBar
                     | ImGuiWindowFlags_NoResize);
        ImDrawList *draw_list = ImGui::GetWindowDrawList();
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                    ImGui::GetIO().Framerate);
        ImGui::DragFloat("Space", &current_space, 10, 200, 1600, "%f");
        ImGui::DragFloat("Gravity", &gravity, 0.5, 0, 1000, "%f");
        ImGui::DragFloat("Radius", &radius, 0.5, 2, 20, "%f");
        ImGui::DragInt("Bodies", &current_bodies, 1, 2, 1000, "%d");
        ImGui::DragFloat("Elapse", &elapse, 0.001, 0.001, 10, "%f");
        ImGui::DragFloat("Max Mass", &current_max_mass, 0.5, 5, 100, "%f");
        ImGui::ColorEdit4("Color", &color.x);
        if (current_space != space || current_bodies != bodies || current_max_mass != max_mass) {
            space = current_space;
            bodies = current_bodies;
            max_mass = current_max_mass;
            delete[] hostX;
            delete[] hostY;
            delete[] hostVX;
            delete[] hostVY;
            delete[] hostAX;
            delete[] hostAY;
            delete[] hostM;
            hostX = new double[bodies];
            hostY = new double[bodies];
            hostVX = new double[bodies];
            hostVY = new double[bodies];
            hostAX = new double[bodies];
            hostAY = new double[bodies];
            hostM = new double[bodies];
            initArray(hostX, hostY, hostM, bodies ,space, max_mass);
            cudaMemcpy(x, hostX, sizeof(double)*bodies, cudaMemcpyHostToDevice);
            cudaMemcpy(y, hostY, sizeof(double)*bodies, cudaMemcpyHostToDevice);
            cudaMemcpy(vx, hostVX, sizeof(double)*bodies, cudaMemcpyHostToDevice);
            cudaMemcpy(vy, hostVY, sizeof(double)*bodies, cudaMemcpyHostToDevice);
            cudaMemcpy(ax, hostAX, sizeof(double)*bodies, cudaMemcpyHostToDevice);
            cudaMemcpy(ay, hostAY, sizeof(double)*bodies, cudaMemcpyHostToDevice);
            cudaMemcpy(m, hostM, sizeof(double)*bodies, cudaMemcpyHostToDevice);
        }
        {   
            struct timeval timeStart, timeEnd;
            double runTime=0;
            const ImVec2 p = ImGui::GetCursorScreenPos();
            gettimeofday(&timeStart, NULL );
            // pool.update_for_tick(elapse, gravity, space, radius);
            // dim3 grid_size;
            // grid_size.x = 1;
            // grid_size.y = 1;
            // grid_size.z = 1;

            // dim3 block_size;
            // block_size.x = 1;
            // block_size.y = 1;
            // block_size.z = 4;
            // int thread_nums = grid_size.x*grid_size.y*grid_size.z*block_size.x*block_size.y*block_size.z;
            int local_bodies = ceil(double(bodies)/thread_num);
            update_for_tick<<<thread_num,1>>>(x,y,vx,vy,ax,ay,m,elapse,gravity, space, radius, local_bodies,bodies);
            cudaDeviceSynchronize();
            cudaMemcpy(hostX, x, sizeof(double)*bodies, cudaMemcpyDeviceToHost);
            cudaMemcpy(hostY, y, sizeof(double)*bodies, cudaMemcpyDeviceToHost);

            gettimeofday( &timeEnd, NULL );
            runTime = (timeEnd.tv_sec - timeStart.tv_sec) + (double)(timeEnd.tv_usec -timeStart.tv_usec)/1000000;
            count_runtime += runTime;
            if(count%1000 == 0 && count != 0){
                printf("runTime for 1000 iteration is  %lf\n", count_runtime);
                count_runtime = 0;
            }
            for (size_t i = 0; i < bodies; ++i) {
                auto xx = p.x + static_cast<float>(hostX[i]);
                auto yy = p.y + static_cast<float>(hostY[i]);
                draw_list->AddCircleFilled(ImVec2(xx, yy), radius, ImColor{color});
            }
        }
        ImGui::End();
        count++;
    });
}
