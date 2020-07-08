#include<iostream>
#include<fstream>
#include<vector>
#include<cmath>
#include<ctime>
#include<omp.h>
#include<random>
#include<thread>
#include<string>
#include<limits>
#include<iomanip>
#include<stdlib.h>

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
std::vector<float> bias;                                        //bias


std::vector<int> inputi;                                        //indices of input neurons
std::vector<int> outputi;                                       //indices of output neurons
std::vector<float> inputsr;                                     //input vector
std::vector<float> outputsr;                                    //output vector

std::vector<double> ppz;                                        //variables for storing backpropgation results                         
std::vector<std::vector<double>> pp1;           
std::vector<std::vector<double>> pp2;            

std::vector<std::vector<std::vector<float>>> Tnn;                 //for storing all blocks of timesteps for tbptt
std::vector<std::vector<float>> Tnnp;                             //container for backpropgation of one block of timesteps
std::vector<std::vector<std::vector<float>>> Ttar;                //for storing all targets for all blocks of timesteps
std::vector<std::vector<int>> Ttari;                              //for storing the index/(timestep) of the targets for each blocks

float weight_cap = 4;                                  //cap on the absolute value of a weight
float bias_cap = 18;                                     //cap on the absolute value of a weight

unsigned long long int Lthreadz;                         //number of cpu threads to run

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
float beta;

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
    float layerbalance = NN.size();//std::sqrt(NN.size()) * log(NN.size()/3) * log(NN.size()/3) * log(NN.size()/3);
    #pragma omp parallel proc_bind(spread)
    {   
        double connectn;
        double connectn2;
        bool it;
        unsigned long long int rrn;
        bool tr;
        long long int avc2;
        long long int avc1;
        double chance1;
        double chance2;
        std::normal_distribution<double> dis(0,rconnect_sdeviation);
        std::uniform_real_distribution<double> tri(0.0,1.0);
        #pragma omp for schedule(nonmonotonic:dynamic)
        for(int y = 0;y < NN.size();y++){
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
            std::normal_distribution<float> Xavier(0,std::sqrt(2.0 / ((W1i[y].size() + W2i[y].size() + connectn + connectn2) * layerbalance)));
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
            rW1i[i].resize(0);
        }
        #pragma omp for
        for(int i = 0 ; i < NN.size() ; i++){
            rW2i[i].resize(0);
        }
        #pragma omp for
        for(int i = 0 ; i < NN.size() ; i++){
            rW1s[i].resize(0);
        }
        #pragma omp for
        for(int i = 0 ; i < NN.size() ; i++){
            rW2s[i].resize(0);
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
}
void prune(float cutoff){
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
        long long it;
        #pragma omp for
        for(unsigned long long int x = 0 ; x < NN.size(); ++x){
            it = 0;
            for(unsigned long long int a = 0; a < W1i[x].size(); a++){
                if(std::abs(W1s[x][a + it]) < cutoff){
                    W1s[x].erase(W1s[x].begin() + a + it);
                    W1i[x].erase(W1i[x].begin() + a + it);
                    --it;
                }
            }
            it = 0;
            for(unsigned long long int a = 0; a < W2i[x].size(); a++){
                if(std::abs(W2s[x][a +it]) < cutoff){
                    W2s[x].erase(W2s[x].begin() + a + it);
                    W2i[x].erase(W2i[x].begin() + a + it);
                    --it;
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
            rW1i[i].resize(0);
        }
        #pragma omp for
        for(int i = 0 ; i < NN.size() ; i++){
            rW2i[i].resize(0);
        }
        #pragma omp for
        for(int i = 0 ; i < NN.size() ; i++){
            rW1s[i].resize(0);
        }
        #pragma omp for
        for(int i = 0 ; i < NN.size() ; i++){
            rW2s[i].resize(0);
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
/*inline double sig(double x, float z = 256){
    long double i;
    long double j;
    i = x / z;
    j = 1 + std::abs(i);
    return i / j;
}*/
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
void L1reg(float regparam){
    #pragma omp parallel for schedule(nonmonotonic:dynamic)
    for(unsigned long long int i = 0; i < W1i.size(); i++){
        #pragma omp simd
        for(unsigned long long int j = 0; j < W1i[i].size(); j++){
            W1s[i][j] -= (regparam * ((W1s[i][j]>0) ? 1:-1)); 
        }
        #pragma omp simd
        for(unsigned long long int j = 0; j < W2i[i].size(); j++){
            W2s[i][j] -= (regparam * ((W2s[i][j]>0) ? 1:-1));
        }
    }
}
void L2reg(float regparam){ 
    #pragma omp parallel for schedule(nonmonotonic:dynamic)
    for(unsigned long long int i = 0; i < W1i.size(); i++){
        #pragma omp simd
        for(unsigned long long int j = 0; j < W1i[i].size(); j++){
            W1s[i][j] -= (regparam * 2 * W1s[i][j]);
        }
        #pragma omp simd
        for(unsigned long long int j = 0; j < W2i[i].size(); j++){
            W2s[i][j] -= (regparam * 2 * W2s[i][j]);
        }
    }
}
void L12reg(float regparam1,float regparam2){
    #pragma omp parallel for schedule(nonmonotonic:dynamic)
    for(unsigned long long int i = 0; i < W1i.size(); i++){
        #pragma omp simd
        for(unsigned long long int j = 0; j < W1i[i].size(); j++){
            W1s[i][j] -= ((regparam2 * 2 * W1s[i][j])+(regparam1 * ((W1s[i][j]>0) ? 1:-1)));
        }
        #pragma omp simd
        for(unsigned long long int j = 0; j < W2i[i].size(); j++){
            W2s[i][j] -= ((regparam2 * 2 * W2s[i][j]) + (regparam1 * ((W2s[i][j]>0) ? 1:-1)));
        }
    }
}
void the_top(int indice){
    Tnnp.resize(Tnn[indice].size());
    float one_minus_beta = 1 - beta;
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
            #pragma omp simd
            for(long long int j = 0; j < Ttar[indice][i].size(); ++j){
                Tnnp[Ttari[indice][i]][outputi[j]] += 2 * (Tnn[indice][Ttari[indice][i]][outputi[j]] - Ttar[indice][i][j]) * dereLU9(Tnn[indice][Ttari[indice][i]][outputi[j]]) * one_minus_beta; //derivative of mean squared error times derivative of reLU
            }
        }
        for(long long int i = layermap.size() - 1; i >= 0 ; i--){
            #pragma omp for schedule(nonmonotonic:dynamic)
            for(long long int j = 0; j < layermap[i].size(); j++){
                acc = 0;
                #pragma omp simd reduction(+:acc)
                for(long long int k = 0 ; k < rW1i[layermap[i][j]].size(); ++k){
                    acc += (Tnnp[Tnnp.size() - 1][rW1i[layermap[i][j]][k]] * rW1s[layermap[i][j]][k]);//<floatflow) ? (Tnnp[Tnnp.size() - 1][rW1i[layermap[i][j]][k]] * rW1s[layermap[i][j]][k] * dTnnp[Tnnp.size() - 1][layermap[i][j]]):floatflow; will fix this some time later
                }
                acc *= (dereLU9(Tnn[indice][Tnnp.size() - 1][layermap[i][j]]) * one_minus_beta);
                #pragma omp atomic //hint(omp_sync_hint_speculative)
                    Tnnp[Tnnp.size() - 1][layermap[i][j]] += acc;
            }
        }
        for(long long int i = Tnnp.size() - 2; i >= 0; --i){
            #pragma omp for schedule(nonmonotonic:dynamic)
            for(long long int j = 0 ; j < NN.size() ; j += 16){
                for(long long int k = j; k < ((j + 16<NN.size()) ? (j + 16) : NN.size()); ++k){
                    //#pragma omp atomic //hint(omp_sync_hint_speculative)
                    Tnnp[i][j] += (Tnnp[i + 1][j] * dereLU9(Tnn[indice][i][j]) * one_minus_beta);
                }
            }
            #pragma omp for schedule(nonmonotonic:dynamic)
            for(long long int j = 0 ; j < NN.size() ; j+= 16){
                for(long long int r = j ; r < ((j + 16 < NN.size()) ? (j + 16) : NN.size()); ++r){
                    acc = 0;
                    #pragma omp simd reduction(+:acc)
                    for(long long int k = 0; k < rW2i[r].size(); ++k){
                        acc += (Tnnp[i + 1][rW2i[r][k]] * rW2s[r][k]);//<floatflow) ? (Tnnp[i + 1][rW2i[j][k]] * rW2s[j][k] * dTnnp[i][j]):floatflow;
                    }
                    acc *= (dereLU9(Tnn[indice][i][r]) * one_minus_beta);
                    //#pragma omp atomic //hint(omp_sync_hint_speculative)
                    Tnnp[i][r] += acc;
                }
            }
            for(long long int j = layermap.size() - 2; j >= 0; --j){
                #pragma omp for schedule(nonmonotonic:dynamic)
                for(long long int k = 0 ; k < layermap[j].size() ; k++){
                    acc = 0;
                    #pragma omp simd reduction(+:acc)
                    for(long long int l = 0 ; l < rW1i[layermap[j][k]].size() ; l++){
                        acc += (Tnnp[i][rW1i[layermap[j][k]][l]] * rW1s[layermap[j][k]][l]);//<floatflow) ? (Tnnp[i][rW1i[layermap[j][k]][l]] * rW1s[layermap[j][k]][l] * dTnnp[i][layermap[j][k]]) : floatflow;
                    }
                    acc *= (dereLU9(Tnn[indice][i][layermap[j][k]]) * one_minus_beta);
                    #pragma omp atomic //hint(omp_sync_hint_speculative)
                        Tnnp[i][layermap[j][k]] += acc;
                }
            }
        }
        #pragma omp for schedule(nonmonotonic:dynamic,16)
        for(long long int i = 0 ; i < Tnnp[0].size(); ++i){
            #pragma omp simd
            for(long long int j = 0 ; j < W1i[i].size() ; j++){
                pp1[i][j] += (Tnn[indice][0][W1i[i][j]] * Tnnp[0][i]);
            }
            #pragma omp atomic
                ppz[i] += Tnnp[0][i];
        }
        for(long long int i = 1; i < Tnnp.size() ; i++){
            #pragma omp for schedule(nonmonotonic:dynamic,16)
            for(long long int j = 0; j < Tnnp[i].size(); j++){
                #pragma omp simd
                for(long long int k = 0 ; k < W2i[j].size(); k++){
                    pp2[j][k] += (Tnn[indice][i - 1][W2i[j][k]] * Tnnp[i][j]);
                }
                #pragma omp simd
                for(long long int k = 0 ; k < W1i[j].size(); k++){
                    pp1[j][k] += (Tnn[indice][i][W1i[j][k]] * Tnnp[i][j]);
                }
                #pragma omp atomic
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
        #pragma omp for schedule(nonmonotonic:dynamic)
        for(unsigned long long int i = 0; i < W1i.size(); i++){
            #pragma omp simd
            for(unsigned long long int j = 0; j < W1i[i].size(); j++){
                //pp1[i][j] = sig(pp1[i][j]);
                pp1[i][j] = reLcap(pp1[i][j]);
            }
            #pragma omp simd
            for(unsigned long long int j = 0; j < W2i[i].size(); j++){
                //pp2[i][j] = sig(pp2[i][j]);
                pp2[i][j] = reLcap(pp2[i][j]);
            }
        } 
        #pragma omp for simd schedule(nonmonotonic:dynamic,16)
        for(unsigned long long int i = 0; i < bias.size(); i++){
            //ppz[i] = sig(ppz[i]);
            ppz[i] = reLcap(ppz[i]);
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
void mario(double wtlearning_rate,double bialearning_rate,float regparam,int Lreg = 0){     
    #pragma omp parallel proc_bind(spread)
    {
        //momentum
        #pragma omp for
        for(unsigned long long int x = 0; x < W1i.size(); x++){
            pp1[x].resize(W1i[x].size());
            #pragma omp simd
            for(unsigned long long int y = 0; y < W1i[x].size(); y++){
                pp1[x][y] *= beta;
            }
        }
        #pragma omp for
        for(unsigned long long int x = 0; x < W2i.size(); x++){
            pp2[x].resize(W2i[x].size());
            #pragma omp simd
            for(unsigned long long int y = 0; y < W2i[x].size(); y++){
                pp2[x][y] *= beta;
            }
        }
        #pragma omp for simd
        for(unsigned long long int x = 0; x < bias.size(); x++){
            ppz[x] *= beta;
        }

    }
    for(unsigned long long int x = 0; x < Tnn.size();x++){
        the_top(x);    
    }
    switch (Lreg)
    {
    case 2:
        L2reg(regparam);
        break;
    case 1:
        L1reg(regparam);
        break;
    default:
        break;
    }
    descent(wtlearning_rate,bialearning_rate);
}
void resetplacehold(){
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

template <typename T> 
void save_param(const T &var,std::ostream &file){ 
    file << std::fixed << std::setprecision(std::numeric_limits<T>::max_digits10)  << var << "\n";
}

template<typename s>            
void save_param(const std::vector<s> &vec, std::ostream &file){
    file << "{" << "\n";
    for (unsigned long long int x = 0; x < vec.size(); ++x){
        save_param(vec[x],file);
    }
    file << "}" << "\n";
}

template<typename T>            
void read_vec(T &, std::istream &){
    std::cout<<"an error has occured when reading the vector"<<std::endl;
    exit(1);
}

template<typename s>            
void read_vec(std::vector<s> &vec, std::istream &file){
    std::string line;
    while(true)
    {
        std::getline(file,line);
        if(line == "{"){
            long long i = vec.size();
            vec.resize(i + 1);
            read_vec(vec[i],file);
        }
        else if(line == "}"){
            return;
        }
        else{
            vec.emplace_back(std::stof(line));
        }
    }
    
}

template<typename s>            
void load_param(std::vector<s> &vec, std::istream &file){
    std::string character;
    while(true)
    {
        std::getline(file,character);
        if(character == "{"){
            read_vec(vec,file);
            return;
        }
        else{
            std::cout<<"an error has occured when reading the vector..."<<std::endl;
            exit(1);
        }
    }
    
}

void savetotxt(){
    std::ofstream textfile("parameters.txt",std::fstream::trunc);
    save_param(W1i,textfile);
    save_param(W1s,textfile);
    save_param(rW1i,textfile);
    save_param(rW1s,textfile);
    save_param(W2i,textfile);
    save_param(W2s,textfile);
    save_param(rW2i,textfile);
    save_param(rW2s,textfile);
    save_param(layermap,textfile);
    save_param(bias,textfile);
    save_param(inputi,textfile);
    save_param(outputi,textfile);
    textfile.close();
}

void loadfromtxt(){
    std::ifstream textfile("parameters.txt");
    load_param(W1i,textfile);
    load_param(W1s,textfile);
    load_param(rW1i,textfile);
    load_param(rW1s,textfile);
    load_param(W2i,textfile);
    load_param(W2s,textfile);
    load_param(rW2i,textfile);
    load_param(rW2s,textfile);
    load_param(layermap,textfile);
    load_param(bias,textfile);
    load_param(inputi,textfile);
    load_param(outputi,textfile);
    textfile.close();
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
void iteration(double app, double bpp,float regparam,int Lreg = 0){
    resetNN();
    int fishaaa = twisting() % maxsteps + 1;
    int lasagne = twisting() % (maxsteps - 1) + 2;
    Tnn[0].resize(lasagne + fishaaa);
    Ttar[0].resize(2);
    Ttari[0].resize(2);
    inputsr[1] = 0;
    for(int i = 0 ; i < fishaaa; i++){
        inputsr[0] = twisting() % 10;
        fire();
        Tnn[0][i].resize(NN.size());
        #pragma omp simd
        for(int j = 0 ; j < NN.size(); j++){
            Tnn[0][0][j] = NN[j];
        }
    }
    inputsr[0] = twisting() % 10;
    inputsr[1] = 9;
    fire();
    Tnn[0][fishaaa].resize(NN.size());
    #pragma omp simd
    for(int j = 0 ; j < NN.size(); j++){
        Tnn[0][fishaaa][j] = NN[j];
    }
    Ttar[0][0].resize(1);
    Ttar[0][0][0] = inputsr[0];
    Ttar[0][1].resize(1);
    Ttar[0][1][0] = inputsr[0];
    Ttari[0][0] = fishaaa;
    Ttari[0][1] = lasagne + fishaaa - 1;
    for(int i = 1; i < lasagne;i++){
        inputsr[0] = twisting() % 10;
        inputsr[1] = 0;
        fire();
        Tnn[0][fishaaa + i].resize(NN.size());
        #pragma omp simd
        for(int j = 0 ; j < NN.size(); j++){
            Tnn[0][fishaaa + i][j] = NN[j];
        }
    }
    inputsr[0] = 0;
    inputsr[1] = 1;
    fire();
    Tnn[0][lasagne + fishaaa - 1].resize(NN.size());
    #pragma omp simd
    for(int j = 0 ; j < NN.size(); j++){
        Tnn[0][lasagne + fishaaa - 1 ][j] = NN[j];
    }    
    mario(app, bpp ,regparam ,Lreg );
    //mario(6 * sig(app,3), 6 * sig(bpp,3),regparam,Lreg);
}
long double terror;
int test(int times = 1){
    terror = 0;
    int avx;
    for(int r = 0 ; r < times; r++){
        resetNN();
        inputsr[0] = twisting() % 10;
        avx = inputsr[0];
        inputsr[1] = 9;
        fire();
        inputsr[1] = 0;
        for(int x = 1 ; x < maxsteps - 1; x++){
            inputsr[0] = twisting() % 10;
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
    int regu = 0;
    float lambda = 0;
    bool fish = 0;
    bool fishy = 0;
    float cut;
    omp_set_dynamic(0);
    std::cout<<"enter number of threads you can/are allowed to run concurrently on CPU"<<std::endl;
    std::cin>>Lthreadz;
    notnum(Lthreadz);
    omp_set_num_threads(Lthreadz);
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            loadfromtxt();
        }
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
            std::cout<<"enter momentum"<<std::endl;
            std::cin>>beta;
            std::cout<<"weight pruning cutoff"<<std::endl;
            std::cin>>cut;
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
                std::cout<<"enter 0 or 1, 0 = no, 1 = yes"<<std::endl;
            }
            while(true){
                int eer;
                std::cout<<"L regularisation ? (0, 1 or 2)"<<std::endl;
                std::cin>>eer;
                if(eer == 1){
                    regu = 1;
                    fishy = 1;
                    break;
                }
                else if(eer == 2){
                    regu = 2;
                    fishy = 1;
                    break;
                }
                else if (eer == 0){
                    fishy = 0;
                    break;
                }
                std::cout<<"enter 0, 1 or 2, 0 = none, 1 = L1, 2 = L2"<<std::endl;
            }
        }
    }
    if(fish){
        new_neuron();
    }
    if(fishy){
        std::cout<<"enter regularisation hyperparameter"<<std::endl;
        std::cin>>lambda;
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
        sync(); 
        prune(cut); 
        resetplacehold();
        for(int i = 0 ; i < 100 ; i ++){
            lrateww = lratew;
            lratebb = lrateb; 
            for(int p = 0; p < cycling; p++){
                lrateww *= deprate;
                lratebb *= deprate;
                iteration(lrateww,lratebb,lambda,regu);
            } 
        }
        std::cout<<"\r"<<t + 1;
    }
    std::cout<<std::endl;
    std::cout<<std::endl;
    std::cout<<"training sets complete ------------"<<std::endl;
    std::cout<<std::endl;
    std::cout<<std::endl;
    std::cout<<"starting evaluation ---------------"<<std::endl;
    std::cout<<std::endl;
    int times = 10000;
    int tr = test(10000);
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
        savetotxt();
    }
    std::cout<<std::endl;
    std::cout<<"session complete ---------------------"<<std::endl;
    return 0;
}


