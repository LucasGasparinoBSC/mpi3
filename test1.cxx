#include <iostream>
#include <cstdlib>
#include <mpi.h>
#include <nvToolsExt.h>
#include <openacc.h>

int main(int argc, char const *argv[])
{
    // Initialize the MPI environment
    int rank, size;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Processor rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Number of processors

    // Get number of GPUs
    int num_gpus = acc_get_num_devices(acc_device_nvidia);
    if (rank == 0) {
        std::cout << "GPUs in the node : " << num_gpus << std::endl;
    }

    // Bind GPU to a rank
    if (num_gpus == size) {
       acc_set_device_num(rank,acc_device_nvidia);
       acc_init(acc_device_nvidia);
       std::cout << "Rank : " << rank << " , GPU : " << acc_get_device_num(acc_device_nvidia) << std::endl;
    }

    // Create buffers
    int bufsize = 10;
    int buf[bufsize];

    nvtxRangePushA("Initialize buffers");
    if (rank == 1) {
        #pragma acc parallel loop
        for (int i = 0; i < bufsize; i++) {
            buf[i] = i;
        }
    } else {
       #pragma acc parallel loop
       for (int i = 0; i < bufsize; i++) {
          buf[i] = 0;
       }
    }
    nvtxRangePop();

    // Create window from buffers
    nvtxRangePushA("Create window");
    MPI_Win win;
    MPI_Win_create(&buf, (MPI_Aint)bufsize*sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    MPI_Win_fence(0, win);
    nvtxRangePop();

    // Rank 0 receives the data from rank 1
    if (rank != 1) {
        nvtxRangePushA("Receive data");
        MPI_Get(&buf, bufsize, MPI_INT, 1, 0, bufsize, MPI_INT, win);
        nvtxRangePop();
    }
    MPI_Win_fence(0, win);

    // All ranks add rankID to the data
    nvtxRangePushA("Add rankID");
    #pragma acc parallel loop
    for (int i = 0; i < bufsize; i++) {
        buf[i] += rank;
    }
    MPI_Win_fence(0, win);
    nvtxRangePop();

    // Print the data
    std::cout << "Rank " << rank << ": ";
    for (int i = 0; i < bufsize; i++) {
        std::cout << buf[i] << " ";
    }
    std::cout << std::endl;

    // Finalize the MPI environment
    MPI_Finalize();
    return 0;
}
