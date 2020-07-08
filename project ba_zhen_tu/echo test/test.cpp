#include<iostream>
#include<fstream>
#include<vector>
#include<cmath>
#include<ctime>
#include<omp.h>
#include<random>
#include<thread>


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

std::random_device rdev;                 
std::mt19937 twisting(rdev());
unsigned long long iterations;
unsigned long long int Lthreadz;                                           //number of cpu threads to run
unsigned long long maxsteps;



inline float reLU9(float x){    
    if(x < 9){  
        return ((x<0)?0:x);   //if (x < 0){return 0;} else{return x;}
    }
    else{
        return 9;
    }
}

void NNout(){
    std::cout<<"{";
    for(int i = 0; i < NN.size(); i++){
        std::cout<<NN[i]<<",";
    }
    std::cout<<"}"<<std::endl;
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
            for(int j = 0; j < ((i + 16<copyNN.size()) ? i + 16:copyNN.size()); j++){
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

void resetNN(){
    #pragma omp for simd schedule(static,16)
    for(int i = 0 ; i < NN.size() ; i++ ){
        NN[i] = 0;
    }
}

long double error;

int test(){
    error = 0;
    int avx;
    for(int r = 0 ; r < iterations; r++){
        resetNN();
        for(int x = 0 ; x < (twisting() % maxsteps); x++){
            inputsr[0] = twisting() % 10;
            inputsr[1] = 0;
            fire();
            //NNout();
        }
        inputsr[0] = twisting() % 10;
        avx = inputsr[0];
        inputsr[1] = 9;
        fire();
        //NNout();
        inputsr[1] = 0;
        for(int x = 1 ; x < maxsteps - 1; x++){
            inputsr[0] = twisting() % 10;
            inputsr[1] = 0;
            fire();
            //NNout();
        }
        inputsr[0] = 0;
        inputsr[1] = 1;
        fire();
        //NNout();
        //std::cout<<"---------------------------"<<std::endl;
        error += (outputsr[0] - avx) * (outputsr[0] - avx);
    }
    error = error/iterations;
    return avx;
}

template<typename nu>
void notnum(nu num){
    if(num == 0){
        std::cout<<"you entered 0 or you didn't enter a number/correct type"<<std::endl;
        exit(EXIT_FAILURE);
    }
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

int main(){
    loadfromtxt();
    NN.resize(W1i.size());
    outputsr.resize(outputi.size());
    inputsr.resize(inputi.size());
    omp_set_dynamic(0);
    std::cout<<"enter number of threads you can/are allowed to run concurrently on CPU"<<std::endl;
    std::cin>>Lthreadz;
    notnum(Lthreadz);
    omp_set_num_threads(Lthreadz);
    std::cout<<"enter number of timesteps"<<std::endl;
    std::cin>>maxsteps;
    notnum(maxsteps);
    std::cout<<"enter iterations to test"<<std::endl;
    std::cin>>iterations;
    notnum(iterations);
    int tr = test();
    std::cout<<"enter number of iterations to test for"<<std::endl;
    std::cout<<std::endl;
    std::cout<<"average echo error = "<<std::fixed<<error<<std::endl;
    std::cout<<"final iteration output "<<std::fixed<<outputsr[0]<<std::endl;
    std::cout<<"final iteration target "<<tr<<std::endl;
}
