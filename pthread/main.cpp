#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <iostream>
#include <cstring>
#include <nbody/body.hpp>
#include <cmath>
#include <cstdlib>
#include <sys/time.h>

using namespace std;

// pthread_mutex_t mutex_p;
pthread_cond_t mutex_threshold;
#define THREADS_NUM 4

template <typename ...Args>
void UNUSED(Args&&... args [[maybe_unused]]) {}

struct Arguments{
    BodyPool *pool;
    double elapse;
    double gravity;
    double position_range;
    double radius;
    int pid;
    int local_bodies;
    int bodies;
};


void *local_process(void *arg_ptr){
    auto arguments = static_cast<Arguments *>(arg_ptr);
    // pthread_mutex_lock(&mutex_p);
    (*arguments->pool).update_for_tick(arguments->elapse, arguments->gravity, 
                                        arguments->position_range, arguments->radius, 
                                        arguments->pid+1, arguments->local_bodies,
                                        arguments-> bodies);
    // pthread_mutex_unlock(&mutex_p);                                        
    delete arguments;
    return nullptr;
}


int main(int argc, char **argv) {
    UNUSED(argc, argv);
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
    int count = 0;
    double count_runtime = 0;
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
            struct timeval timeStart, timeEnd;
            double runTime=0;
            const ImVec2 p = ImGui::GetCursorScreenPos();
            // pool.update_for_tick(elapse, gravity, space, radius);
            gettimeofday(&timeStart, NULL );
            pthread_attr_t attr;
            pthread_attr_init(&attr);
            pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

            pthread_t threads[THREADS_NUM];
            int local_bodies = ceil(double(bodies)/THREADS_NUM);

            for(int m=0; m<THREADS_NUM; m++){
                    pthread_create(&threads[m], nullptr, local_process, new Arguments{
                        .pool = &pool,
                        .elapse = elapse,
                        .gravity = gravity,
                        .position_range = space,
                        .radius = radius,
                        .pid = m,
                        .local_bodies = local_bodies,
                        .bodies = bodies
                    });
            }
            for(auto & n : threads){
                pthread_join(n, nullptr);
            }
            pthread_attr_destroy(&attr);

            gettimeofday( &timeEnd, NULL );
            runTime = (timeEnd.tv_sec - timeStart.tv_sec) + (double)(timeEnd.tv_usec -timeStart.tv_usec)/1000000;
            count_runtime += runTime;
            if(count%1000 == 0 && count != 0){
                printf("runTime for 1000 iteration is  %lf\n", count_runtime);
                count_runtime = 0;
            }

            for (size_t i = 0; i < pool.size(); ++i) {
                auto body = pool.get_body(i);
                auto x = p.x + static_cast<float>(body.get_x());
                auto y = p.y + static_cast<float>(body.get_y());
                draw_list->AddCircleFilled(ImVec2(x, y), radius, ImColor{color});
            }
        }
        ImGui::End();
        count++;
    });
}
