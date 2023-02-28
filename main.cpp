/*
Author: Mohamed Ghanem
Class: ECE4122 A
Last Date Modified: 30/11/2022
Description: Program that uses MPI to compute the steady state heat conduction of a 2D metal plate.
*/
#include <mpi.h>
#include <iostream>
#include <cmath>
#include <getopt.h>
#include <fstream>
#include <chrono>
#include <iomanip>

using namespace std;

int main(int argc, char** argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // rank of processor
    MPI_Comm_size(MPI_COMM_WORLD, &size); // total number of processors
    int n, iter, opt;
    while ((opt = getopt(argc, argv, "n:I:")) != -1)
    {
        switch (opt)
        {
            case 'I':
                iter = stoi(optarg); // iterations
                break;
            case 'n':
                n = stoi(optarg); // interior points
                break;
            default:
                exit(EXIT_FAILURE);
        }
    }
    chrono::time_point<chrono::high_resolution_clock> start, end;
    if (rank == 0)
    {
        start = std::chrono::high_resolution_clock::now(); // start time
    }
    int width = n + 2;
    double *G;
    G = (double*)malloc(width * width * sizeof(double)); // initialize matrix
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < width; j++)
        {
            if (i == 0 || j == 0 || i == width - 1 || j == width - 1)
            {
                if (i == 0 && (j + 1) > ceil(0.3 * width) && (j + 1) <= floor(0.7 * width))
                {
                    G[width * i + j] = 100.0; // set boarder to appropriate value
                }
                else
                {
                    G[width * i + j] = 20.0; // set boarder to appropriate value
                }
            }
            else
            {
                G[width * i + j] = 0.0; // set center to appropriate value
            }
        }
    }
    int elementsPerProcess = ceil(double(width * width) / size); // calculate number of elements calculated per processor
    int subArraySize = (rank + 1) * elementsPerProcess > width * width ? ((width * width) % elementsPerProcess) : elementsPerProcess; // edge case
    double dG[subArraySize]; // sub matrix that will be used to reconstruct conduction matrix
    int revCount[size]; // receive count matrix that determines how many elements to receive from each processor
    int displacements[size]; // displacement matrix that determines the displacement for each processor
    for (int i = 0; i < size ; i++)
    {
        revCount[i] = elementsPerProcess;
        displacements[i] = i * elementsPerProcess;
    }
    revCount[size - 1] = size * elementsPerProcess > width * width ? ((width * width) % elementsPerProcess) : elementsPerProcess; // edge case
    for (int i = 0; i < iter; i++)
    {
        if (rank == 0)
        {
            cout << i << endl;
        }
        for (int j = 0; j < subArraySize ; j++)
        {
            int x = (rank * elementsPerProcess + j) / width; // x index in conduction matrix
            int y = (rank * elementsPerProcess + j) % width; // y index in conduction matrix
            if (x < width - 1 && y < width - 1 && x > 0 && y > 0)
            {
                dG[j] = 0.25 * (G[width * x + y + 1] + G[width * x + y - 1] + G[width * (x - 1) + y] + G[width * (x + 1) + y]); // perform calculation
            }
            else
            {
                dG[j] = G[width * x + y]; // keep border elements unchanged
            }
        }
        // All gather that combines sub matrices from each processor into the conduction matrix
        MPI_Allgatherv(dG, subArraySize, MPI_DOUBLE, G, revCount, displacements,MPI_DOUBLE, MPI_COMM_WORLD);
    }
    if (rank == 0)
    {
        end = std::chrono::high_resolution_clock::now(); // stop time
        chrono::duration<double, milli> ms = end - start; // duration
        cout << "Thin plate calculation took " << ms.count() << " milliseconds." << endl;
        // output matrix to csv file
        ofstream file;
        file.open("finalTemperatures.csv", ios::out);
        for (int i = 0; i < width; i++)
        {
            for (int j = 0; j < width ; j++)
            {
                file << fixed << setprecision(15) << G[width * i + j];
                if (j < width - 1)
                {
                    file << ",";
                }
            }
            file << endl;
        }
        file.close();
    }
    free(G);
    MPI_Finalize();
    return 0;
}