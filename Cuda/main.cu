#include <iostream>
#include <vector>
#include "dev_array.h"
#include "curand.h"
using namespace  std;


int main()
{
    curandGenerator_t curandGenerator;
    curandGenerator(&curandGenerator, CURAND_RNG_PSEUDO_MTGP32);
    curandSetPseudoRandomGeneratorSeed(curandGenerator, 1234ULL);

    const size_t Number_Paths = 100;
    const size_t Number_Steps = 100;
    const float T = 1.0f;


    constexpr size_t N_NORMALS = Number_Paths*Number_Steps;
    const float dt = static_cast<float>(T)/static_cast<float>(Number_Steps);
    float sqrdt = sqrt(dt);

    vector<float>  s(Number_Paths);
    dev_array<float> d_s(Number_Paths);
    dev_array<float> d_normals(N_NORMALS);

    curandGenerateNormal(curandGenerator, d_normals.getData(), N_NORMALS, 0.0f, sqrdt);

    std::cout << "Hello, World!" << std::endl;
    return 0;
}