#include<iostream>
#include<fstream>
#include<vector>
#include<cmath>
#include<ctime>
#include<omp.h>
#include<random>
#include <stdlib.h>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/serialization/vector.hpp>


thread_local std::random_device rdev;                            //this number may not be non deterministic e.g.on mingw before gcc 9.2, be careful
thread_local std::mt19937 twisting(rdev());    

//if you find that there isn't enough precision in long double use mpfr.h for some functions

using namespace std;


/*before you ready your pitchforks , yes I understand that there is probably an excess of global variables,
but many of these variables are may take up large amounts of memory and are used in multiple different functions 
so it would just be easier to have them on the heap, 
also some functions will use way too many arguements
changing our functions to use/accept multiple arguements should be quite simple,
you might want to modify how each function works internally anyways.
another reason is so that I can have all of these variables in a list and group them here so their purpose can be explained to some extent
most of them are constants anyways*/


int NNs;                                                //number of neurons
vector<float> NN;                                       //the Neural Network is represented as a vector


vector<vector<int>> W1i;                                //weights pt1 index of input neuron
vector<vector<int>> W2i;                                //weights pt2 index of input
vector<vector<float>> W1s;                              //the multiplier of the weight
vector<vector<float>> W2s;                              //the multiplier of the weight

vector<vector<int>> rW1i;                               //same connections put in different order for easier concurrent backpropagation
vector<vector<int>> rW2i;                                
vector<vector<float>> rW1s;                              
vector<vector<float>> rW2s;                               

vector<vector<int>> layermap;                           //map of fastlane into layers

vector<float> bias;                                     //bias


float connect_base;                                     //base percentage of available connections that will become new connections with each sync() call
//to add some more randomness into the connecting of neurons (1 + |random gaussian|) * connect_base is used mean of gaussian is 0, |a| means absolute value of a
float rconnect_sdeviation;                              //the standard deviation for the random number (for connections) 
float connect_cap;                                     //the cap on the absolute value of the random number (we don't want any single neuron to start with too many connections)


vector<int> inputi;                                     //vector of input indices
vector<int> outputi;                                    //vector of output indices

int Lthreadz;

//aborts the program if variable is not number/0 is read, (only for a few/not all inputs)
template<typename nu>
void notnum(nu num){
    if(num == 0){
        cout<<"you entered 0 or you didn't enter a number"<<endl;
        exit (EXIT_FAILURE);
    }
}


void syncinit(){
    unsigned long long int list = NNs;
    int itr = 0;
    vector<int> layer = {};
    int ind = 0;
    layermap.emplace_back(layer);
    static vector<int> layertrack;
    static vector<int> neuronindx;
    layertrack.resize(NN.size());
    neuronindx.resize(NN.size());
    #pragma omp parallel num_threads(Lthreadz) proc_bind(spread)
    {
        #pragma omp for
        for(int i = 0; i < neuronindx.size(); i+=16){
            #pragma omp simd
            for(int j = 0; j < ((i + 16<neuronindx.size()) ? i + 16:neuronindx.size()); j++){
                neuronindx[j] = j;
            }
        }
        double connectn;
        double connectn2;
        bool it;
        unsigned long long int rrn;
        bool tr;
        unsigned long long int avc2;
        unsigned long long int avc1;
        double chance1;
        double chance2;
        normal_distribution<double> dis(0,rconnect_sdeviation);
        uniform_real_distribution<double> tri(0.0,1.0);
        #pragma omp master
        {
            tr = false;
            for(int i = 0 ; i < inputi.size(); ++i){
                if(inputi[i] == 0){
                    tr = true;
                    W2i[0].clear();
                    W2s[0].clear();
                    W1i[0].clear();
                    W1s[0].clear();
                    break;
                }
            }
            if(tr){
                ;
            }
            else{
                avc2 = NNs - 1;
                connectn2 = (1 + abs(dis(twisting))) * connect_base;
                connectn2 = (connectn2<connect_cap) ? connectn2 : connect_cap;//min(connectn,connect_cap)
                connectn2 *= avc2; //(NN.size() - (y + 1)) is number of available connections W2
                chance2 = connectn2 - floor(connectn2);
                connectn2 = (chance2>tri(twisting)) ? (floor(connectn2) + 1): floor(connectn2);
                connectn2 = (connectn2<avc2) ? connectn2:avc2;
                normal_distribution<float> Xavier(0,sqrt(2.0/(connectn2 + 1)));
                W2i[0].emplace_back(NNs-1);
                W2s[0].emplace_back(abs(Xavier(twisting)));
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
                            W2i[0].insert(W2i[0].begin() + z ,rrn);
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
        for(long long int y = 1;y < NNs; ++y){
            tr = false;
            for(int i = 0 ; i < inputi.size(); ++i){
                if(inputi[i] == y){
                    tr = true;
                    W2i[y].clear();
                    W2s[y].clear();
                    W1i[y].clear();
                    W1s[y].clear();
                    break;
                }
            }
            if(tr){
                continue;
            }
            avc1 = y - 1;
            avc2 = NNs - y - 1;
            connectn = (1 + abs(dis(twisting))) * connect_base;
            connectn2 = (1 + abs(dis(twisting))) * connect_base;
            connectn = (connectn<connect_cap) ? connectn : connect_cap;//min(connectn,connect_cap)
            connectn2 = (connectn2<connect_cap) ? connectn2 : connect_cap;//min(connectn,connect_cap)
            connectn *=  avc1;  //y - 1 is number of available connections for W1
            connectn2 *= avc2; //(NN.size() - (y + 1)) is number of available connections W2
            chance1 = connectn - floor(connectn);
            chance2 = connectn2 - floor(connectn2);
            connectn = (chance1>tri(twisting)) ? (floor(connectn) + 1): floor(connectn);
            connectn2 = (chance2>tri(twisting)) ? (floor(connectn2) + 1): floor(connectn2);
            connectn = (connectn<avc1) ? connectn:avc1;
            connectn2 = (connectn2<avc2) ? connectn2:avc2;
            normal_distribution<float> Xavier(0,sqrt(2.0/(connectn + connectn2 + 1)));
            W1i[y].emplace_back(y-1);
            W1s[y].emplace_back(abs(Xavier(twisting)));
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
                        W2i[y].insert(W2i[y].begin() + z ,rrn);
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
        for(int i = 0; i < layertrack.size(); i+=16){
            #pragma omp simd
            for(int j = 0; j < ((i + 16<layertrack.size()) ? i + 16:layertrack.size()); j++){
                layertrack[j] = j;
            }
        }
        #pragma omp for
        for(int i = 0 ; i < neuronindx.size() ; ++i ){
            if (layertrack[i] == 0)
            {
                #pragma omp critical
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
            #pragma omp for schedule(nonmonotonic:dynamic,32)
            for(int i = 0 ; i < neuronindx.size(); ++i ){
                for(int j = 0 ; j < layermap[ind].size(); ++j){
                    #pragma omp simd
                    for(int k = 0; k < W1i[i].size(); ++k){
                        if(W1i[i][k] == layermap[ind - 1][j]){
                            --layertrack[neuronindx[i]];
                        }
                    }
                }
                if(layertrack[i] == 0)
                {
                    #pragma omp critical
                    {
                        layermap[ind].emplace_back(neuronindx[i + itr]);
                        neuronindx.erase(neuronindx.begin() + i + itr);
                        --list;
                        --itr;
                    }
                }
            }
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


//reads the input indices
inline void inputscan(){
    int length;
    int ind;
    cout<<"enter number of input neurons"<<endl;
    cin>>length;
    if(length > NNs){
        cout<<"error: not enough neurons, entered number is greater than total number of neurons"<<endl;
        exit (EXIT_FAILURE);
    }
    notnum(length);
    cout<<"enter input neuron indices"<<endl;
    for(int x = 0; x < length; ++x){
        cout<<"enter input neuron index (remember vector indexing starts from 0)"<<endl;
        cin>>ind;
        if((ind >= NNs) || (ind < 0)){
            cout<<"error: index out of range"<<endl;
            exit (EXIT_FAILURE);
        }
        else{
            inputi.insert(inputi.end(),ind);
        }
    }
}


//reads the output indices
inline void outputscan(){
    int length;
    int ind;
    cout<<"enter number of output neurons"<<endl;
    cin>>length;
    if(length > NNs){
        cout<<"error: not enough neurons, entered number is greater than total number of neurons"<<endl;
        exit (EXIT_FAILURE);
    }
    notnum(length);
    cout<<"enter input neuron indices"<<endl;
    for(int x = 0; x < length; ++x){
        cout<<"enter output neuron index (remember vector indexing starts from 0)"<<endl;
        cin>>ind;
        if((ind >= NNs) || (ind < 0)){
            cout<<"error: index out of range"<<endl;
            exit (EXIT_FAILURE);
        }
        else
        {
            outputi.insert(outputi.end(),ind);
        }
    }
}


//creates the binary files
void savebinf(){
    ofstream layermapbin("layermapbin",ofstream::trunc); 
    boost::archive::binary_oarchive  ilayermapbin(layermapbin); 
    ilayermapbin & layermap; 
    ofstream W2ibin("W2i.bin",ofstream::trunc); 
    boost::archive::binary_oarchive  iW2bin(W2ibin); 
    iW2bin & W2i; 
    ofstream W1ibin("W1i.bin",ofstream::trunc); 
    boost::archive::binary_oarchive  iW1bin(W1ibin); 
    iW1bin & W1i;
    ofstream rW2ibin("rW2i.bin",ofstream::trunc); 
    boost::archive::binary_oarchive  riW2bin(rW2ibin); 
    riW2bin & rW2i; 
    ofstream rW1ibin("rW1i.bin",ofstream::trunc); 
    boost::archive::binary_oarchive  riW1bin(rW1ibin); 
    riW1bin & rW1i;
    ofstream W1sbin("W1s.bin",ofstream::trunc); 
    boost::archive::binary_oarchive  sW1bin(W1sbin); 
    sW1bin & W1s; 
    ofstream W2sbin("W2s.bin",ofstream::trunc); 
    boost::archive::binary_oarchive  sW2bin(W2sbin); 
    sW2bin & W2s;
    ofstream rW1sbin("rW1s.bin",ofstream::trunc); 
    boost::archive::binary_oarchive  rsW1bin(rW1sbin); 
    rsW1bin & rW1s; 
    ofstream rW2sbin("rW2s.bin",ofstream::trunc); 
    boost::archive::binary_oarchive  rsW2bin(rW2sbin); 
    rsW2bin & rW2s;
    ofstream biasbin("bias.bin",ofstream::trunc); 
    boost::archive::binary_oarchive  biasesbin(biasbin); 
    biasesbin & bias;
    ofstream inputibin("inputi.bin",ofstream::trunc); 
    boost::archive::binary_oarchive  iinputbin(inputibin); 
    iinputbin & inputi;  
    ofstream outputibin("outputi.bin",ofstream::trunc); 
    boost::archive::binary_oarchive  ioutputbin(outputibin); 
    ioutputbin & outputi;  
}


//creates the xml files
void savexmlf(){ 
    ofstream layermapxml("layermap.xml",ofstream::trunc);
    boost::archive::xml_oarchive ilayermapxml(layermapxml);
    ilayermapxml & BOOST_SERIALIZATION_NVP(layermap); 
    ofstream W2ixml("W2i.xml",ofstream::trunc);  
    boost::archive::xml_oarchive  iW2xml(W2ixml);  
    iW2xml & BOOST_SERIALIZATION_NVP(W2i); 
    ofstream W1ixml("W1i.xml",ofstream::trunc);  
    boost::archive::xml_oarchive  iW1xml(W1ixml);  
    iW1xml & BOOST_SERIALIZATION_NVP(W1i);   
    ofstream W1sxml("W1s.xml",ofstream::trunc);  
    boost::archive::xml_oarchive  sW1xml(W1sxml);  
    sW1xml & BOOST_SERIALIZATION_NVP(W1s);  
    ofstream W2sxml("W2s.xml",ofstream::trunc);  
    boost::archive::xml_oarchive  sW2xml(W2sxml);  
    sW2xml & BOOST_SERIALIZATION_NVP(W2s);  
    ofstream rW2ixml("rW2i.xml",ofstream::trunc);  
    boost::archive::xml_oarchive  riW2xml(rW2ixml);  
    riW2xml & BOOST_SERIALIZATION_NVP(rW2i); 
    ofstream rW1ixml("rW1i.xml",ofstream::trunc);  
    boost::archive::xml_oarchive  riW1xml(rW1ixml);  
    riW1xml & BOOST_SERIALIZATION_NVP(rW1i);   
    ofstream rW1sxml("rW1s.xml",ofstream::trunc);  
    boost::archive::xml_oarchive  rsW1xml(rW1sxml);  
    rsW1xml & BOOST_SERIALIZATION_NVP(rW1s);  
    ofstream rW2sxml("rW2s.xml",ofstream::trunc);  
    boost::archive::xml_oarchive  rsW2xml(rW2sxml);  
    rsW2xml & BOOST_SERIALIZATION_NVP(rW2s);
    ofstream biasxml("bias.xml",ofstream::trunc);  
    boost::archive::xml_oarchive  biasesxml(biasxml);  
    biasesxml & BOOST_SERIALIZATION_NVP(bias);   
    ofstream inputixml("inputi.xml",ofstream::trunc);  
    boost::archive::xml_oarchive  iinputxml(inputixml);  
    iinputxml & BOOST_SERIALIZATION_NVP(inputi); 
    ofstream outputixml("outputi.xml",ofstream::trunc);  
    boost::archive::xml_oarchive  ioutputxml(outputixml);  
    ioutputxml & BOOST_SERIALIZATION_NVP(outputi);   
}


int main(){
    clock_t r = clock();
    omp_set_dynamic(0);
    cout<<"CHECK IF YOU HAVE ENOUGH MEMORY (in heap), always estimate worst case memory usage and have much more than enough memory available."<<endl;
    cout<<"when entering numbers use only decimal fractions and decimal integers, no fractions"<<endl<< "only 1 decimal point is allowed per number, only one - (negative sign) sign is allowed per number"<<endl;
    cout<<"enter number of logical processors/number of threads you are allowed concurrently"<<endl;
    cin>>Lthreadz;
    notnum(Lthreadz);
    omp_set_num_threads(Lthreadz);
    cout<<"enter number of neurons"<<endl;
    cin>>NNs;
    notnum(NNs);
    inputscan();
    outputscan();
    cout<<"enter connect base for initialisation"<<endl;
    cin>>connect_base;
    cout<<"enter rconnectrate standard deviation"<<endl;
    cin>>rconnect_sdeviation;
    cout<<"enter rconnectrate cap "<<endl;
    cin>>connect_cap;
    inputi = {0};
    outputi = {0};
    omp_set_num_threads(Lthreadz);
    clock_t t = clock();
    cout<<"wait..."<<endl;
    vector<float> vec(NNs,0);
    NN = vec;
    bias = vec;
    vector<vector<int>> vec1(NNs);
    W1i = vec1;
    W2i = vec1;
    rW1i = vec1;
    rW2i = vec1;
    vector<vector<float>> vec2(NNs);
    W1s = vec2;
    W2s = vec2;
    rW1s = vec2;
    rW2s = vec2;
    vec.clear();
    vec1.clear();
    vec2.clear();
    syncinit();
    string ttt;
    t = clock() - t;
    double time_elapsed = ((double)t) / CLOCKS_PER_SEC;
    cout << time_elapsed << " seconds to complete initialisation" << endl;
    while(true){
        cout<<"type xml for xml file, bin for binary files or both for both xml and binary files"<<endl;
        cin>>ttt;
        if(ttt == "xml"){
            clock_t r = clock();
            cout<<"wait..."<<endl;
            savexmlf();
            t = t + (clock() - r);
            break;
        }
        else if(ttt == "bin"){
            clock_t r = clock();
            cout<<"wait..."<<endl;
            savebinf();
            t = t + (clock() - r);
            break;
        }
        else if(ttt == "both"){
            clock_t r = clock();
            cout<<"wait..."<<endl;
            savexmlf();
            savebinf();
            t = t + (clock() - r);
            break;
        }
        else{
            cout<<"error: invalid input; enter xml for xml file, bin for binary files (omit the .), or both for copies of both file types"<<endl;
        }
    }
    cout<<"files have been created"<<endl;
    double time_taken = ((double)t) / CLOCKS_PER_SEC;
    cout << time_taken << " seconds to complete all tasks" << endl;
    r = clock() - r;
    time_taken = ((double)r) / CLOCKS_PER_SEC;
    cout << time_taken << " seconds total time" << endl;
    return 0;
}
