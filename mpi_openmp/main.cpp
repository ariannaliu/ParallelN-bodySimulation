#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <nbody/body.hpp>
#include <vector>
#include <cmath>
#include <mpi.h>
#include <iostream>
#include <omp.h>
#include <sys/time.h>


MPI_Status status;
MPI_Request request;
using namespace std;

template <typename ...Args>
void UNUSED(Args&&... args [[maybe_unused]]) {}

int total_size;
#define THREADS_NUM 4

int main(int argc, char **argv) {
    // UNUSED(argc, argv);
    static float gravity = 100;
    static float space = 800;
    static float radius = 5;
    static int bodies = 20;
    static float elapse = 0.001;
    static ImVec4 color = ImVec4(1.0f, 1.0f, 0.4f, 1.0f);
    static float max_mass = 50;
    static float current_space = space;
    static float current_max_mass = max_mass;
    static int current_bodies = bodies;

    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total_size);
    int slave_size = total_size - 1;

    // MPI_Datatype MPIVector;
    // MPI_Type_contiguous(current_bodies, MPI_DOUBLE, &MPIVector);
    // MPI_Type_commit(&MPIVector);

    if(rank == 0){

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
            ImGui::DragInt("Bodies", &current_bodies, 1, 2, 100, "%d");
            ImGui::DragFloat("Elapse", &elapse, 0.001, 0.001, 10, "%f");
            ImGui::DragFloat("Max Mass", &current_max_mass, 0.5, 5, 100, "%f");
            ImGui::ColorEdit4("Color", &color.x);
            if (current_space != space || current_bodies != bodies || current_max_mass != max_mass) {
                space = current_space;
                bodies = current_bodies;
                max_mass = current_max_mass;
                pool = BodyPool{static_cast<size_t>(bodies), space, max_mass};
            }
            MPI_Datatype MPIVector;
            MPI_Type_contiguous(current_bodies, MPI_DOUBLE, &MPIVector);
            MPI_Type_commit(&MPIVector);

            {   
                struct timeval timeStart, timeEnd;
                double runTime=0;
                const ImVec2 p = ImGui::GetCursorScreenPos();
                // pool.update_for_tick(elapse, gravity, space, radius);
                gettimeofday(&timeStart, NULL );
                std::vector<double> c_x = pool.x;
                std::vector<double> c_y = pool.y;
                std::vector<double> c_vx = pool.vx;
                std::vector<double> c_vy = pool.vy;
                std::vector<double> c_ax = pool.ax;
                std::vector<double> c_ay = pool.ay;
                std::vector<double> c_m = pool.m;
                int para[4];
                para[0] = space;
                para[1] = bodies;
                para[2] = max_mass;
                int local_bodies = ceil(double(bodies)/slave_size);
                para[3] = local_bodies;

                float paraf[3];
                paraf[0] = elapse;
                paraf[1] = gravity;
                paraf[2] = radius;
                

                for(int i=0; i<slave_size; i++){
                   MPI_Send(&(para[0]), 4, MPI_INT, (i+1), (i+1)*10+8, MPI_COMM_WORLD);
                   MPI_Send(&(paraf[0]), 3, MPI_INT, (i+1), (i+1)*10+9, MPI_COMM_WORLD);
                   MPI_Send(&(c_x[0]), 1, MPIVector, (i+1), (i+1)*10+1, MPI_COMM_WORLD);
                   MPI_Send(&(c_y[0]), 1, MPIVector, (i+1), (i+1)*10+2, MPI_COMM_WORLD);
                   MPI_Send(&(c_vx[0]), 1, MPIVector, (i+1), (i+1)*10+3, MPI_COMM_WORLD);
                   MPI_Send(&(c_vy[0]), 1, MPIVector, (i+1), (i+1)*10+4, MPI_COMM_WORLD);
                   MPI_Send(&(c_ax[0]), 1, MPIVector, (i+1), (i+1)*10+5, MPI_COMM_WORLD);
                   MPI_Send(&(c_ay[0]), 1, MPIVector, (i+1), (i+1)*10+6, MPI_COMM_WORLD);
                   MPI_Send(&(c_m[0]), 1, MPIVector, (i+1), (i+1)*10+7, MPI_COMM_WORLD);
                }
		        
                MPI_Barrier(MPI_COMM_WORLD);	

                int slave_rank;
                for (int i=0; i<slave_size; i++){
                    MPI_Recv(&(c_x[0]), 1, MPIVector,MPI_ANY_SOURCE, 1, MPI_COMM_WORLD,&status);
                    slave_rank = status.MPI_SOURCE;
                    for (int j=0; j<local_bodies; j++){
                        if((slave_rank-1)*local_bodies + j < bodies){
                            pool.x[(slave_rank-1)*local_bodies + j] = c_x[j];
                        }
                    }
                    MPI_Recv(&(c_y[0]), 1, MPIVector,MPI_ANY_SOURCE, 2, MPI_COMM_WORLD,&status);
                    slave_rank = status.MPI_SOURCE;
                    for (int j=0; j<local_bodies; j++){
                        if((slave_rank-1)*local_bodies + j < bodies){
                            pool.y[(slave_rank-1)*local_bodies + j] = c_y[j];                            
                        }
                    }
                    MPI_Recv(&(c_vx[0]), 1, MPIVector,MPI_ANY_SOURCE, 3, MPI_COMM_WORLD,&status);
                    slave_rank = status.MPI_SOURCE;
                    for (int j=0; j<local_bodies; j++){
                        if((slave_rank-1)*local_bodies + j < bodies){
                            pool.vx[(slave_rank-1)*local_bodies + j] = c_vx[j];                            
                        }
                    }
                    MPI_Recv(&(c_vy[0]), 1, MPIVector,MPI_ANY_SOURCE, 4, MPI_COMM_WORLD,&status);
                    slave_rank = status.MPI_SOURCE;
                    for (int j=0; j<local_bodies; j++){
                        if((slave_rank-1)*local_bodies + j < bodies){
                            pool.vy[(slave_rank-1)*local_bodies + j] = c_vy[j];                            
                        }
                    }
                    MPI_Recv(&(c_ax[0]), 1, MPIVector,MPI_ANY_SOURCE, 5, MPI_COMM_WORLD,&status);
                    slave_rank = status.MPI_SOURCE;
                    for (int j=0; j<local_bodies; j++){
                        if((slave_rank-1)*local_bodies + j < bodies){
                            pool.ax[(slave_rank-1)*local_bodies + j] = c_ax[j];
                        }
                    }
                    MPI_Recv(&(c_ay[0]), 1, MPIVector,MPI_ANY_SOURCE, 6, MPI_COMM_WORLD,&status);
                    slave_rank = status.MPI_SOURCE;
                    for (int j=0; j<local_bodies; j++){
                        if((slave_rank-1)*local_bodies + j < bodies){
                            pool.ay[(slave_rank-1)*local_bodies + j] = c_ay[j];                            
                        }
                    }
                    MPI_Recv(&(c_m[0]), 1, MPIVector,MPI_ANY_SOURCE, 7, MPI_COMM_WORLD,&status);
                    slave_rank = status.MPI_SOURCE;
                    for (int j=0; j<local_bodies; j++){
                        if((slave_rank-1)*local_bodies + j < bodies){
                            pool.m[(slave_rank-1)*local_bodies + j] = c_m[j];                            
                        }
                    }
                }

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
    else{
        // slaves processes
        while(true){
            
            int s_para[4];
            float s_paraf[3];
            MPI_Recv(&(s_para[0]), 4, MPI_INT,0,rank*10+8,MPI_COMM_WORLD, &status);
            MPI_Recv(&(s_paraf[0]), 3, MPI_INT,0,rank*10+9,MPI_COMM_WORLD, &status);
            BodyPool s_pool(static_cast<size_t>(s_para[1]), s_para[0], s_para[2]);
            int local_bodies = s_para[3];
            int s_bodies = s_para[1];
            MPI_Datatype MPILocalVector;
            MPI_Type_contiguous(s_bodies, MPI_DOUBLE, &MPILocalVector);
            MPI_Type_commit(&MPILocalVector);
        
            MPI_Recv(&(s_pool.x[0]), 1, MPILocalVector,0,rank*10+1,MPI_COMM_WORLD, &status);
            MPI_Recv(&(s_pool.y[0]), 1, MPILocalVector,0,rank*10+2,MPI_COMM_WORLD, &status);
            MPI_Recv(&(s_pool.vx[0]), 1, MPILocalVector,0,rank*10+3,MPI_COMM_WORLD, &status);
            MPI_Recv(&(s_pool.vy[0]), 1, MPILocalVector,0,rank*10+4,MPI_COMM_WORLD, &status);
            MPI_Recv(&(s_pool.ax[0]), 1, MPILocalVector,0,rank*10+5,MPI_COMM_WORLD, &status);
            MPI_Recv(&(s_pool.ay[0]), 1, MPILocalVector,0,rank*10+6,MPI_COMM_WORLD, &status);
            MPI_Recv(&(s_pool.m[0]), 1, MPILocalVector,0,rank*10+7,MPI_COMM_WORLD, &status);

#pragma omp parallel num_threads(THREADS_NUM)
            {
            int thread_id = omp_get_thread_num();
            int llocal_bodies = ceil(double(local_bodies)/THREADS_NUM);
            s_pool.update_for_tick(s_paraf[0], s_paraf[1], s_para[0], s_paraf[2], (rank-1)*THREADS_NUM+thread_id+1, llocal_bodies, s_bodies);
            }
            MPI_Isend(&(s_pool.x[0]), 1, MPILocalVector, 0, 1, MPI_COMM_WORLD, &request);
            MPI_Isend(&(s_pool.y[0]), 1, MPILocalVector, 0, 2, MPI_COMM_WORLD, &request);
            MPI_Isend(&(s_pool.vx[0]), 1, MPILocalVector, 0, 3, MPI_COMM_WORLD, &request);
            MPI_Isend(&(s_pool.vy[0]), 1, MPILocalVector, 0, 4, MPI_COMM_WORLD, &request);
            MPI_Isend(&(s_pool.ax[0]), 1, MPILocalVector, 0, 5, MPI_COMM_WORLD, &request);
            MPI_Isend(&(s_pool.ay[0]), 1, MPILocalVector, 0, 6, MPI_COMM_WORLD, &request);
            MPI_Isend(&(s_pool.m[0]), 1, MPILocalVector, 0, 7, MPI_COMM_WORLD, &request); 
	        MPI_Barrier(MPI_COMM_WORLD);         
        }
    }
    MPI_Finalize();
    return 0;
}
