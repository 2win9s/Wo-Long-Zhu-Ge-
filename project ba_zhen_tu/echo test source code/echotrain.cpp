#include<iostream>
#include<fstream>
#include<vector>
#include<cmath>
#include<ctime>
#include<omp.h>
#include<random>
#include<thread>

#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/serialization/vector.hpp>

std::vector<float> NN;                                      
std::vector<std::vector<int>> layermap;                           //map of fastlane into layers


std::vector<std::vector<int>> W1i;                                //weights pt1 index of input neurons
std::vector<std::vector<int>> a1i;                                //list of available input neurons for W1i
std::vector<std::vector<int>> W2i;                                //weights pt2 index of input
std::vector<std::vector<int>> a2i;                                //list of available input neurons for W2i
std::vector<std::vector<float>> W1s;                              //the multiplier of the weight
std::vector<std::vector<float>> W2s;                              //the multiplier of the weight 


std::vector<std::vector<int>> rW1i;                               //same connections put in different order for easier concurrent backpropagation
std::vector<std::vector<int>> rW2i;                                
std::vector<std::vector<float>> rW1s;                              
std::vector<std::vector<float>> rW2s;


std::vector<float> bias;                                     //bias


std::vector<int> inputi;                                     //indices of input neurons
std::vector<int> outputi;                                    //indices of output neurons
std::vector<float> inputsr;                                  //input vector
std::vector<float> outputsr;                                 //output vector

std::vector<double> ppz;                    //variables for storing backpropgation results                         
std::vector<std::vector<double>> pp1;           
std::vector<std::vector<double>> pp2;            


std::vector<std::vector<std::vector<float>>> Tnn;                 //for storing all blocks of timesteps for tbptt
std::vector<std::vector<float>> Tnnp;                             //container for backpropgation of one block of timesteps
std::vector<std::vector<float>> dTnnp;                            //stores the derivatives of the activation function for each neuron
std::vector<std::vector<std::vector<float>>> Ttar;                //for storing all targets for all blocks of timesteps
std::vector<std::vector<int>> Ttari;                              //for storing the index/(timestep) of the targets for each blocks


float weight_cap = 2.5;                                    //cap on the absolute value of a weight
float bias_cap = 200;                                     //cap on the absolute value of a weight


unsigned long long int Lthreadz;                                           //number of cpu threads to run


//threadlocal random number generators7
thread_local std::random_device rdev;                 
thread_local std::mt19937 twisting(rdev());   


float pdeviations;                                      //for pruning weights

int TYBW;                                               //number of sets to train for
int maxsteps;                                           //maximum number of timesteps to train for

float connect_base;                                   //base percentage of available connections that will become new connections with each sync() call
//to add some more randomness into the connecting of neurons (1 + |random gaussian|) * connect_base is used mean of gaussian is 0, |a| means absolute value of a
float rconnect_sdeviation;                            //the standard deviation for the random number (for connections)
float connect_cap;                                    //limit on the percentage of available connections that become connections per call of sync function
double lratew;                                        //weight learning rate
double lrateb;                                        //bias learning rate
int cycling;                                          //number of iterations per set
double deprate;                                       //used as a way to vary the learning rate
float reLUleak;

template<typename r>            
void vecprint1d(r& vec){
    std::cout << "{";
    for (unsigned long long int x = 0; x < vec.size(); x++) {
        if(x != vec.size() - 1){
           std::cout<<std::setprecision(3)<<vec[x]<<", ";
        }
        else{
            std::cout<<vec[x];
        }
    } 
    std::cout<<"}"<<std::endl;
}
template<typename s>            
void vecprint2d(const s& vec){
    std::cout << "{" << std::endl;
    if (!vec.empty()) for (unsigned long long int x = 0; x < vec.size(); x++) {
        std::cout << "{";
        if (!vec[x].empty()) {
            for (unsigned long long int y = 0; y < vec[x].size() - 1; y++) {
                std::cout << vec[x][y] << ", ";
            }
            std::cout << vec[x][vec[x].size() - 1];
        }
        std::cout << "}" << std::endl;
    }
    std::cout << "}" << std::endl;
}
inline long double cregulator(int n,float powlawexp = 1){             
    long double q = std::log(NN.size());           
    long double p = NN.size() - n;
    long double r = p * q;
    long double out = std::pow(r,powlawexp);
    return (1/out) + 0.5;
}
void sync(){
    long long int list = NN.size();
    int itr = 0;
    std::vector<int> layer = {};
    int ind = 0;
    layermap.clear();
    layermap.emplace_back(layer);
    static std::vector<int> layertrack;
    static std::vector<int> neuronindx;
    layertrack.resize(NN.size());
    neuronindx.resize(NN.size());
    #pragma omp parallel proc_bind(spread)
    {   
        double connectn;
        double connectn2;
        bool it;
        unsigned long long int rrn;
        bool tr;
        unsigned long long int avc2;
        unsigned long long int avc1;
        double chance1;
        double chance2;
        std::normal_distribution<double> dis(0,rconnect_sdeviation);
        std::uniform_real_distribution<double> tri(0.0,1.0);
        #pragma omp master
        {
            tr = false;
            for(int i = 0 ; i < inputi.size(); i++){
                if(inputi[i] == 0){
                    tr = true;
                    break;
                }
            }
            if(tr){
                ;
            }
            else
            {
                avc2 = (NN.size() - W2i[0].size() - 1);
                connectn2 = (1 + std::abs(dis(twisting))) * connect_base * cregulator(W2i[0].size());
                connectn2 = (connectn2<connect_cap) ? connectn2 : connect_cap;
                connectn2 *= avc2;
                chance2 = connectn2 - std::floor(connectn2);
                connectn2 = (chance2>tri(twisting)) ? (std::floor(connectn2) + 1): std::floor(connectn2);
                connectn2 =  (connectn2<avc2) ? connectn2:avc2;
                std::normal_distribution<float> Xavier(0,std::sqrt(2.0 / (W2i[0].size() + connectn2)));
                for(unsigned long long int i = 0; i < connectn2;++i){
                    rrn = twisting() % avc2 + 1;
                    it = true;
                    for(unsigned long long int j = 0; j < W2i[0].size(); ++j){
                        if(W2i[0][j] > rrn){
                            break;
                        }
                        ++rrn;
                    }
                    for(unsigned long long int z = 0; z < W2i[0].size();++z){
                        if(rrn < W2i[0][z]){
                            W2i[0].insert(W2i[0].begin() + z , rrn);
                            W2s[0].insert(W2s[0].begin() + z , Xavier(twisting));
                            --avc2;
                            it = false;
                            break;
                        }
                    }
                    if(it){
                        W2i[0].emplace_back(rrn);
                        W2s[0].emplace_back(Xavier(twisting));
                        --avc2;
                    }
                }
            }    
        }
        #pragma omp for schedule(nonmonotonic:dynamic)
        for(int y = 1;y < NN.size();y++){
            tr = false;
            for(int i = 0 ; i < inputi.size(); i++){
                if(inputi[i] == y){
                    tr = true;
                    break;
                }
            }
            if(tr){
                continue;
            }
            avc2 = (NN.size() - y - W2i[y].size() - 1);
            avc1 = (y - 1 - W1i[y].size());
            connectn = (1 + std::abs(dis(twisting))) * connect_base * cregulator(W1i[y].size() + W2i[y].size());
            connectn2 = (1 + std::abs(dis(twisting))) * connect_base * cregulator(W1i[y].size() + W2i[y].size());
            connectn = (connectn<connect_cap) ? connectn : connect_cap;
            connectn2 = (connectn2<connect_cap) ? connectn2 : connect_cap;
            connectn *= avc1;
            connectn2 *= avc2;
            chance1 = connectn - std::floor(connectn);
            chance2 = connectn2 - std::floor(connectn2);
            connectn = (chance1>tri(twisting)) ? (std::floor(connectn) + 1) : std::floor(connectn);
            connectn2 = (chance2>tri(twisting)) ? (std::floor(connectn2) + 1) : std::floor(connectn2);
            connectn =  (connectn<avc1) ? connectn:avc1;
            connectn2 =  (connectn2<avc2) ? connectn2:avc2;
            std::normal_distribution<float> Xavier(0,std::sqrt(2.0 / (W1i[y].size() + W2i[y].size() + connectn + connectn2)));
            for(unsigned long long int i = 0; i < connectn2;++i){
                rrn = twisting() % avc2 + (y + 1);
                it = true;
                for(unsigned long long int j = 0; j < W2i[y].size(); ++j){
                    if(W2i[y][j] > rrn){
                        break;
                    }
                    ++rrn;
                }
                for(unsigned long long int z = 0; z < W2i[y].size();++z){
                    if(rrn < W2i[y][z]){
                        W2i[y].insert(W2i[y].begin() + z , rrn);
                        W2s[y].insert(W2s[y].begin() + z , Xavier(twisting));
                        --avc2;
                        it = false;
                        break;
                    }
                }
                if(it){
                    W2i[y].emplace_back(rrn);
                    W2s[y].emplace_back(Xavier(twisting));
                    --avc2;
                }
            }
            for(unsigned long long int i = 0; i < connectn; ++i){
                rrn = twisting() % avc1;
                it = true;
                for(unsigned long long int j = 0; j < W1i[y].size(); ++j){
                    if(W1i[y][j] > rrn){
                        break;
                    }
                    ++rrn;
                }
                for(unsigned long long int z = 0; z < W1i[y].size();++z){
                    if(rrn < W1i[y][z]){
                        W1i[y].insert(W1i[y].begin() + z ,rrn);
                        W1s[y].insert(W1s[y].begin() + z ,Xavier(twisting));
                        --avc1;
                        it = false;
                        break;
                    }
                }
                if(it){
                    W1i[y].emplace_back(rrn);
                    W1s[y].emplace_back(Xavier(twisting));
                    --avc1;
                }
            }
        }
        #pragma omp for
        for(int i = 0; i < neuronindx.size(); i+=16){
            #pragma omp simd
            for(int j = i; j < ((i + 16<neuronindx.size()) ? i + 16:neuronindx.size()); j++){
                neuronindx[j] = j;
            }
        }
        #pragma omp for
        for(int i = 0; i < layertrack.size(); i+=16){
            #pragma omp simd
            for(int j = i; j < ((i + 16<layertrack.size()) ? i + 16:layertrack.size()); j++){
                layertrack[j] = W1i[j].size();
            }
        }
        #pragma omp for
        for(int i = 0 ; i < NN.size() ; i++){
            rW1i[i].clear();
        }
        #pragma omp for
        for(int i = 0 ; i < NN.size() ; i++){
            rW2i[i].clear();
        }
        #pragma omp for
        for(int i = 0 ; i < NN.size() ; i++){
            rW1s[i].clear();
        }
        #pragma omp for
        for(int i = 0 ; i < NN.size() ; i++){
            rW2s[i].clear();
        }
        #pragma omp sections
        {
            #pragma omp section
            {
                for(int i = 0 ; i < W1i.size() ; i++){
                    for(int j = 0 ; j < W1i[i].size(); j++){
                        rW1i[W1i[i][j]].emplace_back(i);
                    }
                }
            }
            #pragma omp section
            {
                for(int i = 0 ; i < W1i.size() ; i++){
                    for(int j = 0 ; j < W1i[i].size(); j++){
                        rW1s[W1i[i][j]].emplace_back(W1s[i][j]);
                    }
                }
            }
            #pragma omp section
            {
                for(int i = 0 ; i < W2i.size() ; i++){
                    for(int j = 0 ; j < W2i[i].size(); j++){
                        rW2i[W2i[i][j]].emplace_back(i);
                    }
                }
            }
            #pragma omp section
            {
                for(int i = 0 ; i < W2i.size() ; i++){
                    for(int j = 0 ; j < W2i[i].size(); j++){
                        rW2s[W2i[i][j]].emplace_back(W2s[i][j]);
                    }
                }
            }
        } 
        #pragma omp single
        {
            for(int i = 0 ; i < layertrack.size() ; ++i ){
                if (layertrack[i] == 0)
                {
                    layermap[ind].emplace_back(neuronindx[i + itr]);
                    neuronindx.erase(neuronindx.begin() + i + itr);
                    --list;
                    --itr;
                }      
            }
        }
        while(list > 0){
            #pragma omp single
            {
                layermap.emplace_back(layer);
                ++ind;
                itr = 0;
            }
            #pragma omp for
            for(int i = 0 ; i < neuronindx.size(); ++i ){
                for(int j = 0 ; j < layermap[ind - 1].size(); ++j){
                    #pragma omp simd
                    for(int k = 0; k < W1i[neuronindx[i]].size(); ++k){
                        if(W1i[neuronindx[i]][k] == layermap[ind - 1][j]){
                            --layertrack[neuronindx[i]];
                        }
                    }
                }
            }
            #pragma omp single
            {
                for(int i = 0 ; i < neuronindx.size(); ++i ){
                    if(layertrack[i] == 0)
                    {
                            layermap[ind].emplace_back(neuronindx[i + itr]);
                            neuronindx.erase(neuronindx.begin() + i + itr);
                            --list;
                            --itr;
                    }
                }
            }
        }
    }
    //vecprint2d(layermap);
    //vecprint2d(layermap);
}
void syncprune(){ 
    unsigned long long int list = NN.size();
    int itr = 0;
    std::vector<int> layer = {};
    int ind = 0;
    layermap.clear();
    layermap.emplace_back(layer);
    static std::vector<int> layertrack;
    static std::vector<int> neuronindx;
    layertrack.resize(NN.size());
    neuronindx.resize(NN.size());
    #pragma omp parallel
    {
        #pragma omp for
        for(int i = 0; i < neuronindx.size(); i+=16){
            #pragma omp simd 
            for(int j = i; j < ((i + 16<neuronindx.size()) ? i + 16:neuronindx.size()); j++){
                neuronindx[j] = j;
            }
        }
        double mean;
        double variance;
        float stdeviation;
        float cutoff;
        long long int it;
        int max;
        #pragma omp for schedule(nonmonotonic:dynamic)
        for(long long int x = 0; x < NN.size(); ++x){
            unsigned long long int population = W1i[x].size() + W2i[x].size();
            mean = 0;
            variance = 0;
            max = 0;
            /*
            #pragma omp simd reduction(+:mean)
            for(unsigned long long int y = 0; y < W1s[x].size();y++){
                max = (max > std::abs(W1s[x][y]) ) ? max : std::abs(W1s[x][y]);
            }
            #pragma omp simd reduction(+:mean)
            for(unsigned long long int y = 0; y < W2s[x].size();y++){
                max = (max > std::abs(W2s[x][y]) ) ? max : std::abs(W2s[x][y]);
            }
            */
            #pragma omp simd reduction(+:mean)
            for(unsigned long long int y = 0; y < W1s[x].size();y++){
                mean += std::abs(W1s[x][y]);
            }
            #pragma omp simd reduction(+:mean)
            for(unsigned long long int y = 0; y < W2s[x].size();y++){
                mean += std::abs(W2s[x][y]);
            }
            mean = mean / population;
            #pragma omp simd reduction(+:variance)
            for(unsigned long long int z = 0; z < W1s[x].size();z++){
                variance += (std::abs(W1s[x][z]) - mean) * (std::abs(W1s[x][z]) - mean);
            }
            #pragma omp simd reduction(+:variance)
            for(unsigned long long int z = 0; z < W2s[x].size();z++){
                variance += (std::abs(W2s[x][z]) - mean) * (std::abs(W2s[x][z]) - mean);
            }
            variance = variance / population;
            stdeviation = std::sqrt(variance);
            cutoff = mean - (stdeviation * pdeviations);
            it = 0;
            for(unsigned long long int a = 0; a < W1i[x].size(); a++){
                if(std::abs(W1s[x][it]) < cutoff){
                    W1s[x].erase(W1s[x].begin() + it);
                    W1i[x].erase(W1i[x].begin() + it);
                }
                else
                {
                    ++it;   
                }
            }
            it = 0;
            for(unsigned long long int a = 0; a < W2i[x].size(); a++){
                if(std::abs(W2s[x][it]) < cutoff){
                    W2s[x].erase(W2s[x].begin() + it);
                    W2i[x].erase(W2i[x].begin() + it);
                }
                else
                {
                    ++it;   
                }
            }
        }
        #pragma omp for
        for(int i = 0; i < layertrack.size(); i+=16){
            #pragma omp simd
            for(int j = i; j < ((i + 16<layertrack.size()) ? i + 16:layertrack.size()); j++){
                layertrack[j] = W1i[j].size();
            }
        }
        #pragma omp single
        {
            for(int i = 0 ; i < layertrack.size() ; ++i ){
                if (layertrack[i] == 0)
                {
                    layermap[ind].emplace_back(neuronindx[i + itr]);
                    neuronindx.erase(neuronindx.begin() + i + itr);
                    --list;
                    --itr;
                }      
            }
        }
        while(list > 0){
            #pragma omp single
            {
                layermap.emplace_back(layer);
                ++ind;
                itr = 0;
            }
            #pragma omp for
            for(int i = 0 ; i < neuronindx.size(); ++i ){
                for(int j = 0 ; j < layermap[ind - 1].size(); ++j){
                    #pragma omp simd
                    for(int k = 0; k < W1i[neuronindx[i]].size(); ++k){
                        if(W1i[neuronindx[i]][k] == layermap[ind - 1][j]){
                            --layertrack[neuronindx[i]];
                        }
                    }
                }
            }
            #pragma omp single
            {
                for(int i = 0 ; i < neuronindx.size(); ++i ){
                    if(layertrack[i] == 0)
                    {
                            layermap[ind].emplace_back(neuronindx[i + itr]);
                            neuronindx.erase(neuronindx.begin() + i + itr);
                            --list;
                            --itr;
                    }
                }
            }
        }
        #pragma omp for
        for(int i = 0 ; i < NN.size() ; i++){
            rW1i[i].clear();
        }
        #pragma omp for
        for(int i = 0 ; i < NN.size() ; i++){
            rW2i[i].clear();
        }
        #pragma omp for
        for(int i = 0 ; i < NN.size() ; i++){
            rW1s[i].clear();
        }
        #pragma omp for
        for(int i = 0 ; i < NN.size() ; i++){
            rW2s[i].clear();
        }
        #pragma omp sections
        {
            #pragma omp section
            {
                for(int i = 0 ; i < W1i.size() ; i++){
                    for(int j = 0 ; j < W1i[i].size(); j++){
                        rW1i[W1i[i][j]].emplace_back(i);
                    }
                }
            }
            #pragma omp section
            {
                for(int i = 0 ; i < W1i.size() ; i++){
                    for(int j = 0 ; j < W1i[i].size(); j++){
                        rW1s[W1i[i][j]].emplace_back(W1s[i][j]);
                    }
                }
            }
            #pragma omp section
            {
                for(int i = 0 ; i < W2i.size() ; i++){
                    for(int j = 0 ; j < W2i[i].size(); j++){
                        rW2i[W2i[i][j]].emplace_back(i);
                    }
                }
            }
            #pragma omp section
            {
                for(int i = 0 ; i < W2i.size() ; i++){
                    for(int j = 0 ; j < W2i[i].size(); j++){
                        rW2s[W2i[i][j]].emplace_back(W2s[i][j]);
                    }
                }
            }
        } 
    }
}
inline float reLU(float x){
    return (x<0)?0:x;   //if (x < 0){return 0;} else{return x;}
}
inline float reLU9(float x){    
    if(x < 9){  
        return ( x < 0 ) ? 0 : x;   //if (x < 0){return 0;} else{return x;}
    }
    else{
        return 9;
    }
}
inline void fire(){
    static std::vector<float> copyNN = NN;
    #pragma omp parallel proc_bind(spread)
    {
        float accumulate;
        #pragma omp for simd schedule(nonmonotonic:dynamic,16)
        for(int i = 0; i < inputi.size(); i++){
            NN[inputi[i]] = inputsr[i];
        }
        #pragma omp single
        {
            copyNN.resize(NN.size());
        }


        #pragma omp for
        for(int i = 0; i < copyNN.size(); i+=16){
            #pragma omp simd
            for(int j = i; j < ((i + 16<copyNN.size()) ? i + 16:copyNN.size()); j++){
                copyNN[j] = NN[j];
            }
        }


        #pragma omp for schedule(nonmonotonic:dynamic,16) 
        for(unsigned long long int x = 0; x < NN.size() ; x++){
            accumulate = 0; 
            #pragma omp simd reduction(+:accumulate)
            for(int z = 0; z < W2i[x].size();z++){
                accumulate += (copyNN[W2i[x][z]] * W2s[x][z]); 
            }
            accumulate += bias[x];
            NN[x] += accumulate;
        }

        #pragma omp for simd
        for(int i = 0 ; i < layermap[0].size() ; ++i){
          NN[layermap[0][i]] = reLU9(NN[layermap[0][i]]);
        }
        
        for(int i = 1 ; i < layermap.size() ; ++i){
            #pragma omp for schedule(nonmonotonic:dynamic)
            for(int j = 0 ; j < layermap[i].size() ; ++j){
                accumulate = 0;
                #pragma omp simd reduction(+:accumulate)
                for(int k = 0 ; k < W1i[layermap[i][j]].size() ; k++){
                    accumulate += (NN[W1i[layermap[i][j]][k]] * W1s[layermap[i][j]][k]);
                }
                accumulate = reLU9(accumulate + NN[layermap[i][j]]);
                #pragma omp atomic write//hint(omp_sync_hint_speculative)
                    NN[layermap[i][j]] = accumulate;
            }
        }

        #pragma omp for simd schedule(nonmonotonic:dynamic,16)
        for(int q = 0; q < outputi.size(); q++){
            outputsr[q] = NN[outputi[q]];
        }
    }
}
inline float dereLU(float x, float leak = 0.01){
   return (x > 0) ? 1:leak;    //if(x>0){return 1;}else{return 0;}   
}
inline float dereLU9(float x, float leak = 0.01){
    if(x >= 9){
        return leak;
    }
    else{
        return (x > 0) ? 1:leak;   //if(x>0){return 1;}else{return 0;}   
    }
}
inline double sig(double x, float z=32){
    long double i;
    long double j;
    i = x / z;
    j = 1 + std::abs(i);
    return i / j;
}
inline double reLcap(double x, float cap = 2048){
    if(x > 0){
        if(x > cap){
            return cap;
        }
        else{
            return x;
        }
    }
    else{
        if(x > (cap * -1)){
            return x;
        }
        else{
            return cap;
        }
    }
}
void the_top(int indice){
    Tnnp.resize(Tnn[indice].size());
    #pragma omp parallel
    {
        float acc;
        #pragma omp for schedule(nonmonotonic:dynamic)
        for(long long int i = 0 ; i < Tnnp.size() ; i++){
            Tnnp[i].resize(Tnn[indice][i].size());
            #pragma omp simd
            for(long long int j = 0 ; j < Tnnp[i].size() ; j++){
                Tnnp[i][j] = 0;
            }
        }
        #pragma omp for schedule(nonmonotonic:dynamic)
        for(unsigned long long int i = 0 ; i < Ttari[indice].size() ; i++ ){
            float m = 1/Ttar[indice][i].size();
            #pragma omp simd
            for(long long int j = 0; j < Ttar[indice][i].size(); ++j){
                Tnnp[Ttari[indice][i]][outputi[j]] += 2 * (Tnn[indice][Ttari[indice][i]][outputi[j]] - Ttar[indice][i][j]) * dTnnp[Ttari[indice][i]][outputi[j]] * m; //derivative of mean squared error times derivative of reLU
            }
        }
        for(long long int i = layermap.size() - 2; i >= 0 ; i--){
            #pragma omp for schedule(nonmonotonic:dynamic)
            for(long long int j = 0; j < layermap[i].size(); j++){
                acc = 0;
                #pragma omp simd reduction(+:acc)
                for(long long int k = 0 ; k < rW1i[layermap[i][j]].size(); ++k){
                    acc += (Tnnp[Tnnp.size() - 1][rW1i[layermap[i][j]][k]] * rW1s[layermap[i][j]][k] * dTnnp[Tnnp.size() - 1][layermap[i][j]]);//<floatflow) ? (Tnnp[Tnnp.size() - 1][rW1i[layermap[i][j]][k]] * rW1s[layermap[i][j]][k] * dTnnp[Tnnp.size() - 1][layermap[i][j]]):floatflow; will fix this some time later
                }
                #pragma omp atomic //hint(omp_sync_hint_speculative)
                    Tnnp[Tnnp.size() - 1][layermap[i][j]] += acc;
            }
        }
        for(long long int i = Tnnp.size() - 2; i >= 0; --i){
            #pragma omp for schedule(nonmonotonic:dynamic)
            for(long long int j = 0 ; j < NN.size() ; j++){
                acc = 0;
                #pragma omp simd reduction(+:acc)
                for(long long int k = 0; k < rW2i[j].size(); ++k){
                    acc += (Tnnp[i + 1][rW2i[j][k]] * rW2s[j][k] * dTnnp[i][j]);//<floatflow) ? (Tnnp[i + 1][rW2i[j][k]] * rW2s[j][k] * dTnnp[i][j]):floatflow;
                }
                #pragma omp atomic //hint(omp_sync_hint_speculative)
                    Tnnp[i][j] += acc;
            }
            for(long long int j = layermap.size() - 2; j >= 0; --j){
                #pragma omp for schedule(nonmonotonic:dynamic)
                for(long long int k = 0 ; k < layermap[j].size() ; k++){
                    acc = 0;
                    #pragma omp simd reduction(+:acc)
                    for(long long int l = 0 ; l < rW1i[layermap[j][k]].size() ; l++){
                        acc += (Tnnp[i][rW1i[layermap[j][k]][l]] * rW1s[layermap[j][k]][l] * dTnnp[i][layermap[j][k]]);//<floatflow) ? (Tnnp[i][rW1i[layermap[j][k]][l]] * rW1s[layermap[j][k]][l] * dTnnp[i][layermap[j][k]]) : floatflow;
                    }
                    #pragma omp atomic //hint(omp_sync_hint_speculative)
                        Tnnp[i][layermap[j][k]] += acc;
                }
            }
        }
        #pragma omp for schedule(nonmonotonic:dynamic,16)
        for(long long int i = 0 ; i < Tnnp[0].size(); ++i){
            for(long long int j = 0 ; j < W1i[i].size() ; j++){
                pp1[i][j] += (Tnn[indice][0][W1i[i][j]] * Tnnp[0][i]);
            }
            ppz[i] += Tnnp[0][i];
        }
        for(long long int i = 1; i < Tnnp.size() ; i++){
            #pragma omp for schedule(nonmonotonic:dynamic,16)
            for(long long int j = 0; j < Tnnp[i].size(); j++){
                for(long long int k = 0 ; k < W2i[j].size(); k++){
                    pp2[j][k] += (Tnn[indice][i - 1][W2i[j][k]] * Tnnp[i][j]);
                }
                for(long long int k = 0 ; k < W1i[j].size(); k++){
                    pp1[j][k] += (Tnn[indice][i][W1i[j][k]] * Tnnp[i][j]);
                }
                ppz[j] += Tnnp[i][j];
            }
        }
    }
}
inline void descent(float wlearn_cap,float blearn_cap){    
    static std::vector<int> itr(NN.size(),0);
    itr.resize(NN.size());
    #pragma omp parallel proc_bind(spread)
    {
        /*#pragma omp for nowait
        for(unsigned long long int i = 0; i < W1s.size() ; i++){
            for(unsigned long long int x = 0; x < W1s[i].size(); x += 16){ //this is such that a slice of p2[0][i] that is 32 long should be able to remain in SIMD register or at least cache until we are completely done with it, 
                for(unsigned long long int j = 1; j < p1.size(); j++){     //At least 1KB L1 cache to not get a lot of L1 cache misses(in this day and age if you don't have 1KB L1, what the hell are you running this on?).  
                    #pragma omp simd
                    for(unsigned long long int k = x; k < ((x + 16<W1s[i].size()) ? (x+16):W1s[i].size()); k++){
                        p1[0][i][k] += p1[j][i][k];
                    }
                }
            }
            for(unsigned long long int x = 0; x < W2s[i].size(); x += 16){ 
                for(unsigned long long int j = 1; j < p2.size(); j++){
                    #pragma omp simd
                    for(unsigned long long int k = x; k < ((x + 16<W2s[i].size()) ? (x+16):W2s[i].size()); k++){
                        p2[0][i][k] += p2[j][i][k];
                    }
                }
            }
        }
        for(unsigned long long int k = 0; k < bias.size();k+=16){
            for(unsigned long long int i = 1; i < p1.size() ; i++){ 
                #pragma omp for simd                     
                for(unsigned long long int j = 0; j < ((k + 16<bias.size()) ? (k+16):bias.size());j++){
                    pz[0][j] += pz[i][j];
                }
            }
        }*/
        #pragma omp for schedule(nonmonotonic:dynamic)
        for(unsigned long long int i = 0; i < W1i.size(); i++){
            #pragma omp simd
            for(unsigned long long int j = 0; j < W1i[i].size(); j++){
                pp1[i][j] = sig(pp1[i][j]);
            }
            #pragma omp simd
            for(unsigned long long int j = 0; j < W2i[i].size(); j++){
                pp2[i][j] = sig(pp2[i][j]);
            }
        } 
        #pragma omp for simd schedule(nonmonotonic:dynamic,16)
        for(unsigned long long int i = 0; i < bias.size(); i++){
            //std::cout<<"before"<<ppz[i]<<std::endl;
            ppz[i] = sig(ppz[i]);
            //std::cout<<"after"<<ppz[i]<<std::endl;
        }
        #pragma omp for schedule(nonmonotonic:dynamic)
        for(unsigned long long int i = 0; i < W1i.size(); i++){
            #pragma omp simd
            for(unsigned long long int j = 0; j < W1i[i].size(); j++){
                W1s[i][j] -= (pp1[i][j] * wlearn_cap);
                W1s[i][j] =  (W1s[i][j]<weight_cap)?W1s[i][j]:weight_cap;   //W1s[i][j] = min(W1s[i][j],weight_cap);
                W1s[i][j] =  (W1s[i][j]>(-1*weight_cap))?W1s[i][j]:(-1*weight_cap);//W2s[i][j] = max(W2s[i][j],-1 * weight_cap);
            }
            #pragma omp simd
            for(unsigned long long int j = 0; j < W2i[i].size(); j++){
                W2s[i][j] -= (pp2[i][j] * wlearn_cap);
                W2s[i][j] =  (W2s[i][j]<weight_cap)?W2s[i][j]:weight_cap;   //W2s[i][j] = min(W2s[i][j],weight_cap);
                W2s[i][j] =  (W2s[i][j]>(-1*weight_cap))?W2s[i][j]:(-1*weight_cap);//W2s[i][j] = max(W2s[i][j],-1 * weight_cap);
            }
        } 
        #pragma omp for simd schedule(nonmonotonic:dynamic,16)
        for(unsigned long long int i = 0; i < bias.size(); i++){
            bias[i] -= (ppz[i] * blearn_cap);
            //std::cout<<"change"<<ppz[i] * blearn_cap<<std::endl;
            //std::cout<<"bias"<<bias[i]<<std::endl;
            bias[i] =  (bias[i]<bias_cap)?bias[i]:bias_cap;   //W2s[i][j] = min(W2s[i][j],weight_cap);
            bias[i] =  (bias[i]>(-1*bias_cap))?bias[i]:(-1*bias_cap);//W2s[i][j] = max(W2s[i][j],-1 * weight_cap);
        }
        #pragma omp for simd schedule(static,16)
        for(int i = 0 ; i < itr.size() ; i++){
            itr[i] = 0;
        }
        for(int i = 0 ; i < W2i.size() ; i++){
            #pragma omp for simd
            for(int j = 0 ; j < W2i[i].size(); j++){
                rW2s[W2i[i][j]][itr[W2i[i][j]]] = W2s[i][j];
                ++itr[W2i[i][j]];
            }
        }
        #pragma omp for simd schedule(static,16)
        for(int i = 0 ; i < itr.size() ; i++){
            itr[i] = 0;
        }
        for(int i = 0 ; i < W1i.size() ; i++){
            #pragma omp for simd
            for(int j = 0 ; j < W1i[i].size(); j++){
                rW1s[W1i[i][j]][itr[W1i[i][j]]] = W1s[i][j];
                ++itr[W1i[i][j]];
            }
        }
        #pragma omp for simd schedule(static,16)
        for(int i = 0 ; i < inputi.size(); ++i){
            bias[inputi[i]] = 0;
        }
    }
}
void mario(double wtlearning_rate,double bialearning_rate){     
    pp1.resize(W1i.size());
    pp2.resize(W2i.size());
    ppz.resize(bias.size());
    #pragma omp parallel proc_bind(spread)
    {
        #pragma omp for
        for(unsigned long long int x = 0; x < W1i.size(); x++){
            pp1[x].resize(W1i[x].size());
            #pragma omp simd
            for(unsigned long long int y = 0; y < W1i[x].size(); y++){
                pp1[x][y] = 0;
            }
        }
        #pragma omp for
        for(unsigned long long int x = 0; x < W2i.size(); x++){
            pp2[x].resize(W2i[x].size());
            #pragma omp simd
            for(unsigned long long int y = 0; y < W2i[x].size(); y++){
                pp2[x][y] = 0;
            }
        }
        #pragma omp for simd
        for(unsigned long long int x = 0; x < bias.size(); x++){
            ppz[x] = 0;
        }
    }
    for(unsigned long long int x = 0; x < Tnn.size();x++){
        #pragma omp single
        {
            dTnnp.resize(Tnn[x].size()); 
        } 
        #pragma omp parallel for
        for(unsigned long long int i = 0; i < Tnn[x].size(); i++){
            dTnnp[i].resize(Tnn[x][i].size());
            #pragma omp simd
            for(unsigned long long int j = 0 ; j < Tnn[x][i].size(); j++){
                dTnnp[i][j] = dereLU9(Tnn[x][i][j],reLUleak);
            }
        }
        the_top(x);    
    }  
    descent(wtlearning_rate,bialearning_rate);
}
template<typename nu>
void notnum(nu num){
    if(num == 0){
        std::cout<<"you entered 0 or you didn't enter a number/correct type"<<std::endl;
        std::exit(EXIT_FAILURE);
    }
}
template<typename s>  
void vec2dsize(const s& vec) {
    std::cout<<"main vector size "<<vec.size()<<std::endl;
    std::cout << "{";
    if (!vec.empty()) for (int x = 0; x < vec.size(); x++) {
            std::cout<<"{"<<vec[x].size()<<"}";
    }
    std::cout << "}" <<std::endl;
}
void loadparam(){
    std::ifstream layermapxml("layermap.xml");
    boost::archive::xml_iarchive ilayermapxml(layermapxml);
    ilayermapxml & BOOST_SERIALIZATION_NVP(layermap);   
    std::ifstream W2ixml("W2i.xml");  
    boost::archive::xml_iarchive  iW2xml(W2ixml);  
    iW2xml & BOOST_SERIALIZATION_NVP(W2i); 
    std::ifstream W1ixml("W1i.xml");  
    boost::archive::xml_iarchive  iW1xml(W1ixml);  
    iW1xml & BOOST_SERIALIZATION_NVP(W1i);   
    std::ifstream W1sxml("W1s.xml");  
    boost::archive::xml_iarchive  sW1xml(W1sxml);  
    sW1xml & BOOST_SERIALIZATION_NVP(W1s);  
    std::ifstream W2sxml("W2s.xml");  
    boost::archive::xml_iarchive  sW2xml(W2sxml);  
    sW2xml & BOOST_SERIALIZATION_NVP(W2s); 
    std::ifstream rW2ixml("rW2i.xml");  
    boost::archive::xml_iarchive  riW2xml(rW2ixml);  
    riW2xml & BOOST_SERIALIZATION_NVP(rW2i); 
    std::ifstream rW1ixml("rW1i.xml");  
    boost::archive::xml_iarchive  riW1xml(rW1ixml);  
    riW1xml & BOOST_SERIALIZATION_NVP(rW1i);   
    std::ifstream rW1sxml("rW1s.xml");  
    boost::archive::xml_iarchive  rsW1xml(rW1sxml);  
    rsW1xml & BOOST_SERIALIZATION_NVP(rW1s);  
    std::ifstream rW2sxml("rW2s.xml");  
    boost::archive::xml_iarchive  rsW2xml(rW2sxml);  
    rsW2xml & BOOST_SERIALIZATION_NVP(rW2s);
    std::ifstream biasxml("bias.xml");  
    boost::archive::xml_iarchive  biasesxml(biasxml);  
    biasesxml & BOOST_SERIALIZATION_NVP(bias);  
    std::ifstream inputixml("inputi.xml");  
    boost::archive::xml_iarchive  iinputxml(inputixml);  
    iinputxml & BOOST_SERIALIZATION_NVP(inputi);
    std::ifstream outputixml("outputi.xml");  
    boost::archive::xml_iarchive  ioutputxml(outputixml);  
    ioutputxml & BOOST_SERIALIZATION_NVP(outputi);  
}
void saveparam(){
    std::ofstream layermapxml("layermap.xml",std::ofstream::trunc);
    boost::archive::xml_oarchive ilayermapxml(layermapxml);
    ilayermapxml & BOOST_SERIALIZATION_NVP(layermap); 
    std::ofstream W2ixml("W2i.xml",std::ofstream::trunc);  
    boost::archive::xml_oarchive  iW2xml(W2ixml);  
    iW2xml & BOOST_SERIALIZATION_NVP(W2i); 
    std::ofstream W1ixml("W1i.xml",std::ofstream::trunc);  
    boost::archive::xml_oarchive  iW1xml(W1ixml);  
    iW1xml & BOOST_SERIALIZATION_NVP(W1i);   
    std::ofstream W1sxml("W1s.xml",std::ofstream::trunc);  
    boost::archive::xml_oarchive  sW1xml(W1sxml);  
    sW1xml & BOOST_SERIALIZATION_NVP(W1s);  
    std::ofstream W2sxml("W2s.xml",std::ofstream::trunc);  
    boost::archive::xml_oarchive  sW2xml(W2sxml);  
    sW2xml & BOOST_SERIALIZATION_NVP(W2s);  
    std::ofstream rW2ixml("rW2i.xml",std::ofstream::trunc);  
    boost::archive::xml_oarchive  riW2xml(rW2ixml);  
    riW2xml & BOOST_SERIALIZATION_NVP(rW2i); 
    std::ofstream rW1ixml("rW1i.xml",std::ofstream::trunc);  
    boost::archive::xml_oarchive  riW1xml(rW1ixml);  
    riW1xml & BOOST_SERIALIZATION_NVP(rW1i);   
    std::ofstream rW1sxml("rW1s.xml",std::ofstream::trunc);  
    boost::archive::xml_oarchive  rsW1xml(rW1sxml);  
    rsW1xml & BOOST_SERIALIZATION_NVP(rW1s);  
    std::ofstream rW2sxml("rW2s.xml",std::ofstream::trunc);  
    boost::archive::xml_oarchive  rsW2xml(rW2sxml);  
    rsW2xml & BOOST_SERIALIZATION_NVP(rW2s);
    std::ofstream biasxml("bias.xml",std::ofstream::trunc);  
    boost::archive::xml_oarchive  biasesxml(biasxml);  
    biasesxml & BOOST_SERIALIZATION_NVP(bias);   
    std::ofstream inputixml("inputi.xml",std::ofstream::trunc);  
    boost::archive::xml_oarchive  iinputxml(inputixml);  
    iinputxml & BOOST_SERIALIZATION_NVP(inputi); 
    std::ofstream outputixml("outputi.xml",std::ofstream::trunc);  
    boost::archive::xml_oarchive  ioutputxml(outputixml);  
    ioutputxml & BOOST_SERIALIZATION_NVP(outputi);    
}
void resetNN(){
    #pragma omp for simd schedule(static,16)
    for(int i = 0 ; i < NN.size() ; i++ ){
        NN[i] = 0;
    }
}
void NNout(){
    std::cout<<"{";
    for(int i = 0; i < NN.size(); i++){
        std::cout<<NN[i]<<",";
    }
    std::cout<<"}"<<std::endl;
}
void new_neuron(){
    static std::vector<int> f = {};
    static std::vector<float> ftt = {};
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            NN.emplace_back(0);
        }
        #pragma omp section
        {
            W1i.emplace_back(f);
        }
        #pragma omp section
        {
            W2i.emplace_back(f);
        }
        #pragma omp section
        {
            rW1i.emplace_back(f);
        }
        #pragma omp section
        {
            rW2i.emplace_back(f);
        }
        #pragma omp section
        {
            W2s.emplace_back(ftt);
        }
        #pragma omp section
        {
            W1s.emplace_back(ftt);
        }
        #pragma omp section
        {
            rW2s.emplace_back(ftt);
        }
        #pragma omp section
        {
            rW1s.emplace_back(ftt);
        }
        #pragma omp section
        {
            layermap[0].emplace_back(NN.size()- 1);
        }
        #pragma omp section
        {
            bias.emplace_back(0);
        }
    }
}
void act3(double app, double bpp){
    resetNN();
    static std::vector<float> targ(1,0);
    int lasagne = (twisting() % (maxsteps - 14)) + 15;
    int truncate = twisting() % 8 + 2;
    Tnn[0].resize(truncate);
    Ttar[0].resize(1);
    Ttari[0].resize(1);

    inputsr[0] = twisting() % 10;
    targ[0] = inputsr[0];
    inputsr[1] = 0;
    fire();
    Ttar[0][0].resize(1);
    Ttar[0][0][0] = targ[0];
    for(int i = 1; i < lasagne - (truncate - 1);i++){
        inputsr[0] = twisting() % 10;
        inputsr[1] = 0;
        fire();
    }    
    for(int i = 0;i < truncate - 1;i++){
        inputsr[0] = twisting() % 10;
        inputsr[1] = 0;
        fire();
        Tnn[0][i].resize(NN.size());
        #pragma omp simd
        for(int j = 0 ; j < NN.size(); j++){
            Tnn[0][i][j] = NN[j];
        }
    }
    inputsr[0] = 0;
    inputsr[1] = 1;
    fire();
    Tnn[0][truncate - 1].resize(NN.size());
    #pragma omp simd
    for(int j = 0 ; j < NN.size(); j++){
        Tnn[0][truncate - 1][j] = NN[j];
    }
    Ttari[0][0] = truncate - 1;

    mario(6 * sig(app,3), 6 * sig(bpp,3));
}
void act2(double app, double bpp){
    resetNN();
    static std::vector<float> targ(1,0);
    int lasagne = (twisting() % (maxsteps/2)) + maxsteps - (maxsteps/2) + 1;
    int truncate = twisting() % 3 + 1;
    Tnn[0].resize(truncate);
    Ttar[0].resize(1);
    Ttari[0].resize(1);

    inputsr[0] = twisting() % 10;
    targ[0] = inputsr[0];
    inputsr[1] = 0;
    fire();
    Ttar[0][0].resize(1);
    Ttar[0][0][0] = targ[0];
    for(int i = 1; i < lasagne - (truncate - 1);i++){
        inputsr[0] = twisting() % 10;
        inputsr[1] = 0;
        fire();
    }    
    for(int i = 0;i < truncate - 1;i++){
        inputsr[0] = twisting() % 10;
        inputsr[1] = 0;
        fire();
        Tnn[0][i].resize(NN.size());
        #pragma omp simd
        for(int j = 0 ; j < NN.size(); j++){
            Tnn[0][i][j] = NN[j];
        }
    }
    inputsr[0] = 0;
    inputsr[1] = 1;
    fire();
    Tnn[0][truncate - 1].resize(NN.size());
    #pragma omp simd
    for(int j = 0 ; j < NN.size(); j++){
        Tnn[0][truncate - 1][j] = NN[j];
    }
    Ttari[0][0] = truncate - 1;

    mario(6 * sig(app,3), 6 * sig(bpp,3));
}
void act1(double app, double bpp){
    resetNN();
    static std::vector<float> targ(1,0);
    int lasagne = (twisting() % (maxsteps/2)) + maxsteps - (maxsteps/2) + 1;
    Tnn[0].resize(lasagne);
    Ttar[0].resize(1);
    Ttari[0].resize(1);
    inputsr[0] = twisting() % 10;
    targ[0] = inputsr[0];
    inputsr[1] = 0;
    fire();
    Tnn[0][0].resize(NN.size());
    #pragma omp simd
    for(int j = 0 ; j < NN.size(); j++){
        Tnn[0][0][j] = NN[j];
    }
    Ttar[0][0].resize(1);
    Ttar[0][0][0] = targ[0];

    for(int i = 1; i < lasagne;i++){
        inputsr[0] = twisting() % 10;
        inputsr[1] = 0;
        fire();
        Tnn[0][i].resize(NN.size());
        #pragma omp simd
        for(int j = 0 ; j < NN.size(); j++){
            Tnn[0][i][j] = NN[j];
        }
    }
    inputsr[0] = 0;
    inputsr[1] = 1;
    fire();
    Tnn[0][lasagne - 1].resize(NN.size());
    #pragma omp simd
    for(int j = 0 ; j < NN.size(); j++){
        Tnn[0][lasagne - 1][j] = NN[j];
    }
    Ttari[0][0] = lasagne - 1;
    mario(6 * sig(app,3), 6 * sig(bpp,3));
}
long double terror;
int test(int times = 10){
    terror = 0;
    int avx;
    for(int r = 0 ; r < times; r++){
        resetNN();
        inputsr[0] = twisting() % 10;
        avx = inputsr[0];
        inputsr[1] = 0;
        fire();
        for(int x = 1 ; x < maxsteps - 1; x++){
            inputsr[0] = twisting() % 10;
            inputsr[1] = 0;
            fire();
        }
        inputsr[0] = 0;
        inputsr[1] = 1;
        fire();
        terror += (outputsr[0] - avx) * (outputsr[0] - avx);
    }
    terror = terror/times;
    return avx;
}
int main(){
    bool fish = 0;
    omp_set_dynamic(0);
    std::cout<<"enter number of threads you can/are allowed to run concurrently on CPU"<<std::endl;
    std::cin>>Lthreadz;
    notnum(Lthreadz);
    omp_set_num_threads(Lthreadz);
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            std::cout<<"enter max number of timesteps, at least 4"<<std::endl;
            std::cin>>maxsteps;
            notnum(maxsteps);
            std::cout<<"enter number of training sets to complete"<<std::endl;
            std::cin>>TYBW;
            notnum(TYBW);
            std::cout<<"enter number of iterations in each set"<<std::endl;
            std::cin>>cycling;
            notnum(cycling);
            std::cout<<"enter weights learning rate"<<std::endl;
            std::cin>>lratew;
            notnum(lratew);
            std::cout<<"enter bias learning rate"<<std::endl;
            std::cin>>lrateb;
            notnum(lrateb);
            std::cout<<"enter learning 'decay' rate"<<std::endl;
            std::cin>>deprate;
            notnum(deprate);
            std::cout<<"enter 'reLUleak' "<<std::endl;
            std::cin>>reLUleak;
            std::cout<<"weight standard deviations cutoff (below mean)"<<std::endl;
            std::cin>>pdeviations;
            notnum(pdeviations);
            std::cout<<"enter base connectrate"<<std::endl;
            std::cin>>connect_base;
            std::cout<<"standard deviation of connectrate multiplier"<<std::endl;
            std::cin>>rconnect_sdeviation;
            std::cout<<"connect rate cap"<<std::endl;
            std::cin>>connect_cap;
            std::cout<<"wait..."<<std::endl;
            while(true){
                int eer;
                std::cout<<"extra neuron ? (0 or 1)"<<std::endl;
                std::cin>>eer;
                if(eer == 1){
                    fish = 1;
                    break;
                }
                else if (eer == 0){
                    fish = 0;
                    break;
                }
                std::cout<<"you didn't enter 0 or 1, 0 = no, 1 = yes"<<std::endl;
            }
        }
        #pragma omp section
        {
            loadparam();
        }
    }
    if(fish){
        new_neuron();
        std::cout<<"new neuron created"<<std::endl;
    }
    #pragma omp parallel 
    {
        #pragma omp sections
        {
            #pragma omp section
            {
                inputsr.resize(inputi.size());
            }
            #pragma omp section
            {
                outputsr.resize(outputi.size());
            }
            #pragma omp section
            {
                Tnn.resize(1);
            }
            #pragma omp section
            {
                Ttar.resize(1);
            }
            #pragma omp section
            {
                Ttari.resize(1);
            }
            #pragma omp section
            {
                NN.resize(W1i.size());
            }
        }
    }
    double lrateww;
    double lratebb;
    std::cout<<std::endl;
    std::cout<<"sets out of "<<TYBW<<" completed:"<<std::endl;
    std::cout<<"0"<<std::flush;
    for(int t = 0; t < TYBW; t++){
        syncprune();
        sync(); 
        std::cout<<std::endl;
        lrateww = lratew;
        lratebb = lrateb;//.005;   
        for(int p = 0; p < cycling; p++){
            lrateww *= deprate;
            lratebb *= deprate;
            if(maxsteps < 22){
                act1(lrateww,lratebb);
            }
            else if(maxsteps < 50){
                act2(lrateww,lratebb);
            }
            else{
                act3(lrateww,lratebb);
            }
        }
        std::cout<<"\r"<<t + 1<<std::flush;
        test();
        if(terror == 0){
            std::cout<<"below error threshold"<<std::endl;
            break;
        }
    }
    std::cout<<std::endl;
    std::cout<<std::endl;
    std::cout<<"training sets complete ------------"<<std::endl;
    std::cout<<std::endl;
    std::cout<<std::endl;
    std::cout<<"starting evaluation ---------------"<<std::endl;
    std::cout<<std::endl;
    int times = 1000;
    int tr = test(1000);
    std::cout<<std::endl;
    std::cout<<"bias current values"<<std::endl;
    vecprint1d(bias);
    std::cout<<"neural network current values"<<std::endl;
    NNout();
    std::cout<<"W1 connections"<<std::endl;
    vec2dsize(W1i);
    std::cout<<"W2 connections"<<std::endl;
    vec2dsize(W2i);
    std::cout<<std::endl;
    std::cout<<std::endl;
    std::cout<<"average echo squared error("<<times<<" iterations) = "<<std::fixed<<terror<<std::endl;
    std::cout<<"final iteration output "<<std::fixed<<outputsr[0]<<std::endl;
    std::cout<<"final iteration target "<<tr<<std::endl;
    std::cout<<std::endl;
    std::cout<<std::endl;
    while(true){
        int eer;
        std::cout<<"save new parameters ? (0 or 1)"<<std::endl;
        std::cin>>eer;
        if(eer == 1){
            fish = 1;
                break;
        }
        else if (eer == 0){
            fish = 0;
            break;
        }
        std::cout<<"you didn't enter 0 or 1, 0 = no, 1 = yes"<<std::endl;
    }
    if(fish){
        std::cout<<"saving new parameters..."<<std::endl;
        std::cout<<"WARNING DO NOT STOP THE PROCESS, OR ELSE ALL PROGRESS WILL BE LOST!!!-------------"<<std::endl;
        saveparam();
    }
    std::cout<<std::endl;
    std::cout<<"session complete ---------------------"<<std::endl;
    return 0;
}
