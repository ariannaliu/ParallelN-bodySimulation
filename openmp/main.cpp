#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <iostream>
#include <cstring>
#include <nbody/body.hpp>
#include <cmath>
#include <cstdlib>
#include <omp.h>
#include <sys/time.h>

using namespace std;

// #define THREADS_NUM 4
// omp_lock_t writelock;

template <typename ...Args>
void UNUSED(Args&&... args [[maybe_unused]]) {}




int main(int argc, char **argv) {
    // UNUSED(argc, argv);
    int THREADS_NUM;
    if (argc < 2){
        // if user did not provide the size of array
        // the defualt value is set to be 100
        THREADS_NUM = 4;
    }else{
        THREADS_NUM = atoi(argv[1]);
    }
    // omp_init_lock(&writelock);
    static float gravity = 100;
    static float space = 800;
    static float radius = 5;
    static int bodies = 200;
    static float elapse = 0.001;
    static ImVec4 color = ImVec4(1.0f, 1.0f, 0.4f, 1.0f);
    static float max_mass = 50;
    static float current_space = space;
    static float current_max_mass = max_mass;
    static int current_bodies = bodies;
    // int count = 0;
    // double count_runtime = 0;
    BodyPool pool(static_cast<size_t>(bodies), space, max_mass);
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
            pool = BodyPool{static_cast<size_t>(bodies), space, max_mass};
        }
        {
            // struct timeval timeStart, timeEnd;
            // double runTime=0;
            const ImVec2 p = ImGui::GetCursorScreenPos();
            // #pragma omp single
            // gettimeofday(&timeStart, NULL );
            {
            omp_set_num_threads(THREADS_NUM);
            pool.update_for_tick(elapse,gravity,space,radius, bodies);
            }
            #pragma omp single
            for (size_t i = 0; i < pool.size(); ++i) {
                auto body = pool.get_body(i);
                auto x = p.x + static_cast<float>(body.get_x());
                auto y = p.y + static_cast<float>(body.get_y());
                draw_list->AddCircleFilled(ImVec2(x, y), radius, ImColor{color});
            }
        }
        ImGui::End();
        // count++;
    });
    // omp_destroy_lock(&writelock);
}
