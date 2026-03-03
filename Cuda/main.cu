#include <iostream>
#include <vector>
#include "curand.h"
#include  "thrust/host_vector.h"
#include  "thrust/device_vector.h"

using namespace  std;
using namespace thrust;


int main()
{
    curandGenerator_t curandGenerator;
    curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_MTGP32);
    curandSetPseudoRandomGeneratorSeed(curandGenerator, 1234ULL);

    constexpr size_t Number_Paths = 100;
    constexpr size_t Number_Steps = 100;
    constexpr float T = 1.0f;


    constexpr size_t N_NORMALS = Number_Paths*Number_Steps;
    const float dt = static_cast<float>(T)/static_cast<float>(Number_Steps);
    float sqrdt = sqrt(dt);

    vector<float>  s(Number_Paths);
    host_vector<float> d_s(Number_Paths);
    host_vector<float> d_normals(Number_Steps);

    curandGenerateNormal(curandGenerator, d_normals.data(), N_NORMALS, 0.0f, sqrdt);

    std::cout << "Hello, World!" << std::endl;
    return 0;
}