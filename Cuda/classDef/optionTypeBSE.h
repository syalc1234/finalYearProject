//
// Created by jayma on 31/03/2026.
//

#ifndef CUDA_OPTIONTYPEBSE_H
#define CUDA_OPTIONTYPEBSE_H


class optionTypeBSE
{
public:
    float sigma ;
    float risk_free_rate ;
    float time_to_exp ;
    float strike_K ;
    float spatial_step ;
    float current_price ;
    float N;

    optionTypeBSE()
    {
        sigma = 0;
        risk_free_rate = 0;
        time_to_exp = 0;
        strike_K = 0;
        spatial_step = 0;
        current_price = 0;
        N = 0;
    }

    optionTypeBSE(float sigma, float risk_free_rate, float time_to_exp, float strike_K, float spatial_step, float current_price, float N)
    {
        this->sigma = sigma ;
        this->risk_free_rate = risk_free_rate ;
        this->time_to_exp = time_to_exp ;
        this->strike_K = strike_K ;
        this->spatial_step = spatial_step ;
        this->current_price = current_price ;
        this->N = N ;
    }
};


#endif //CUDA_OPTIONTYPEBSE_H