//
// Created by jayma on 31/03/2026.
//

#ifndef CUDA_OPTIONTYPE_H
#define CUDA_OPTIONTYPE_H


class optionTypeMC
{
public:
    float s0;
    float mu;
    float sigma;
    float K;
    float r;
    int Number_Paths ;
    int Number_Steps ;
    float T;

    optionTypeMC()
    {
        s0 = 100;
        mu = 0.2;
        sigma = 0.2;
        K = 120;
        r = 0.1;
        Number_Paths = 100000;
        Number_Steps = 1000;
        T = 1.0f;
    }
    optionTypeMC( float s0, float mu, float sigma, float K, float r, float Number_Paths, float Number_Steps, float T)
    {
        this->s0 = s0;
        this->mu = mu;
        this->sigma = sigma;
        this->K = K;
        this->r = r;
        this->Number_Paths = Number_Paths;
        this->Number_Steps = Number_Steps;
        this->T = T;
    }
};


#endif //CUDA_OPTIONTYPE_H