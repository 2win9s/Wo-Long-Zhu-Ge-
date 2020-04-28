/*
Copyright (c) 2020 2win9s permission to copy without fee all or part of this material is granted.
IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY
compile and run/use the code at your own risk, run/compile/use the code appropriately
This code is not meant for proper real world use so please don't judge the quality of code too harshly.
*/
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


vector<vector<int>> W1i;                                //weights pt1 index of input neurons
vector<vector<int>> a1i;                                //list of available input neurons for W1i


vector<vector<int>> W2i;                                //weights pt2 index of input
vector<vector<int>> a2i;                                //list of available input neurons for W2i

vector<vector<float>> W1s;                              //the multiplier of the weight
vector<vector<float>> W2s;                              //the multiplier of the weight 


vector<float> bias;                                     //bias


float connect_base;                                     //a percentage of available connections that will become new connections with each sync() call
//to add some more randomness into the connecting of neurons (1 + |random gaussian|) * connect_base is used mean of gaussian is 0, |a| means absolute value of a
float rconnect_sdeviation;                              //the standard deviation for the random number (for connections) 
float rconnect_cap;                                     //the cap on the absolute value of the random number (we don't want any single neuron to start with too many connections)


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


// function initialises list of available connects for each neuron (a1i and a2i), must be called before syncinit()
void afill(){ 
    #pragma omp parallel proc_bind(spread)
    {
        #pragma omp for simd 
        for(int x = 0;x < a1i.size() - 1;x++){
            for(int y = 0; y < x - 1; y++){
                a1i[x].emplace_back(y);
            }
        }
        #pragma omp for simd nowait
        for(int x = 1; x < a2i.size();x++){
            for(int y = x + 1; y < NNs; y++){
                a2i[x].emplace_back(y);
            }
        }
        #pragma omp single
        {
            for(int i = 1; i < NNs - 1;i++){
                a2i[0].emplace_back(i);
            }
        }
    }
}

//function for staring the weights, this function makes sure that each neuron is connected to the one behind it, the first neuron is connected to the last;
//this ensures that information will flow through all the neurons
void weight_start(){
    normal_distribution<float> distribution(0,1);
    #pragma omp for simd nowait proc_bind(spread)
    for(int i = 1;i < NNs;i++){
        float r = distribution(twisting);
        W1i[i].emplace_back(i - 1);
        W1s[i].emplace_back(r);
    }
    double r = distribution(twisting);
    W2i[0].emplace_back(NNs - 1);
    W2s[0].emplace_back(r);
} 


void syncinit(){
    #pragma omp parallel num_threads(Lthreadz) proc_bind(spread)
    {
        double randnm;
        double chance;
        double connectn;
        int ng;
        float in;
        normal_distribution<double> dis(0,rconnect_sdeviation);
        uniform_real_distribution<double> tri(0.0,1.0);
        #pragma omp for
        for(int y = 1;y < NNs;y++){
            randnm = dis(twisting);
            randnm = (abs(randnm)<rconnect_cap) ? abs(randnm):rconnect_cap; //min(abs(randnm),rconnect_cap)
            connectn = (1 + randnm) * (y - 1) * connect_base;
            if(connectn > 1){
            connectn = floor(connectn);
            }
            else{
                chance = tri(twisting);
                connectn  = (connectn >= chance) ? 1:0;                     //if(connectn >= chance) connectn = 1, else connectn = 0
            }
            connectn =  (connectn<a1i[y].size()) ? connectn:a1i[y].size();  //min(connectn,a1i[y].size())
            normal_distribution<float> al(0,sqrt(2.0 / ((W1i[y].size() + connectn)*2)));
            for(int i = 0; i < connectn; i++){
                ng = twisting() % a1i[y].size();
                W1i[y].emplace_back(a1i[y][ng]);
                in = al(twisting); 
                W1s[y].emplace_back(in);
                a1i[y].erase(a1i[y].begin() + ng);
            }
        }
        #pragma omp for
        for(int y = 0;y < NNs - 1;y++){
            randnm = dis(twisting);
            randnm = (abs(randnm)<rconnect_cap) ? abs(randnm):rconnect_cap; //min(abs(randnm),rconnect_cap)
            connectn = (1 + randnm) * (NNs - (y + 1)) * connect_base;
            if(connectn > 1){
            connectn = floor(connectn);
            }
            else{
                chance = tri(twisting);
                connectn  = (connectn >= chance) ? 1:0;                     //if(connectn >= chance) connectn = 1, else connectn = 0
            }
            connectn =  (connectn<a2i[y].size()) ? connectn:a2i[y].size();  //min(connectn,a2i[y].size())
            normal_distribution<float> al(0,sqrt(2.0 / ((W2i[y].size() + connectn)*2)));
            for(int i = 0; i < connectn; i++){
                ng = twisting() % a2i[y].size();
                W2i[y].emplace_back(a2i[y][ng]);
                in = al(twisting); 
                W2s[y].emplace_back(in);
                a2i[y].erase(a2i[y].begin() + ng);
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
    cout<<"enter input neuron indices"<<endl<<"must start with lowest index"<<endl<<"e.g. if your list is {1,2} wait for prompt type 1, hit enter, wait for next prompt then type 2 ..."<<endl;
    for(int x = 0; x < length; x++){
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
    cout<<"enter input neuron indices"<<endl<<"must start with lowest index"<<endl<<"e.g. if your list is {1,2} wait for prompt type 1, hit enter, wait for next prompt then type 2 ..."<<endl;
    for(int x = 0; x < length; x++){
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
    ofstream a2ibin("a2i.bin",ofstream::trunc); 
    boost::archive::binary_oarchive  ia2bin(a2ibin); 
    ia2bin << a2i; 
    a2i.clear();
    ofstream a1ibin("a1i.bin",ofstream::trunc); 
    boost::archive::binary_oarchive  ia1bin(a1ibin); 
    a1i.clear();
    ia1bin << a1i; 
    ofstream W2ibin("W2i.bin",ofstream::trunc); 
    boost::archive::binary_oarchive  iW2bin(W2ibin); 
    W2i.clear();
    iW2bin << W2i; 
    ofstream W1ibin("W1i.bin",ofstream::trunc); 
    boost::archive::binary_oarchive  iW1bin(W1ibin); 
    iW1bin << W1i;
    W1i.clear();
    ofstream W1sbin("W1s.bin",ofstream::trunc); 
    boost::archive::binary_oarchive  sW1bin(W1sbin); 
    sW1bin << W1s; 
    W1s.clear();
    ofstream W2sbin("W2s.bin",ofstream::trunc); 
    boost::archive::binary_oarchive  sW2bin(W2sbin); 
    sW2bin << W2s;
    W2s.clear();
    ofstream biasbin("bias.bin",ofstream::trunc); 
    boost::archive::binary_oarchive  biasesbin(biasbin); 
    biasesbin << bias;
    bias.clear();
    ofstream inputibin("inputi.bin",ofstream::trunc); 
    boost::archive::binary_oarchive  iinputbin(inputibin); 
    iinputbin << inputi;  
    inputi.clear();
    ofstream outputibin("outputi.bin",ofstream::trunc); 
    boost::archive::binary_oarchive  ioutputbin(outputibin); 
    ioutputbin << outputi;  
    outputi.clear();
}


//creates the xml files
void savexmlf(){ 
    ofstream a2ixml("a2i.xml",ofstream::trunc);  
    boost::archive::xml_oarchive  ia2xml(a2ixml);  
    ia2xml << BOOST_SERIALIZATION_NVP(a2i);  
    a2i.clear();
    ofstream a1ixml("a1i.xml",ofstream::trunc);  
    boost::archive::xml_oarchive  ia1xml(a1ixml);  
    ia1xml << BOOST_SERIALIZATION_NVP(a1i); 
    a1i.clear(); 
    ofstream W2ixml("W2i.xml",ofstream::trunc);  
    boost::archive::xml_oarchive  iW2xml(W2ixml);  
    iW2xml << BOOST_SERIALIZATION_NVP(W2i); 
    W2i.clear();
    ofstream W1ixml("W1i.xml",ofstream::trunc);  
    boost::archive::xml_oarchive  iW1xml(W1ixml);  
    iW1xml << BOOST_SERIALIZATION_NVP(W1i);   
    W1i.clear();
    ofstream W1sxml("W1s.xml",ofstream::trunc);  
    boost::archive::xml_oarchive  sW1xml(W1sxml);  
    sW1xml << BOOST_SERIALIZATION_NVP(W1s);  
    W1s.clear();
    ofstream W2sxml("W2s.xml",ofstream::trunc);  
    boost::archive::xml_oarchive  sW2xml(W2sxml);  
    sW2xml << BOOST_SERIALIZATION_NVP(W2s);
    W2s.clear();  
    ofstream biasxml("bias.xml",ofstream::trunc);  
    boost::archive::xml_oarchive  biasesxml(biasxml);  
    biasesxml << BOOST_SERIALIZATION_NVP(bias);  
    bias.clear(); 
    ofstream inputixml("inputi.xml",ofstream::trunc);  
    boost::archive::xml_oarchive  iinputxml(inputixml);  
    iinputxml << BOOST_SERIALIZATION_NVP(inputi);
    inputi.clear(); 
    ofstream outputixml("outputi.xml",ofstream::trunc);  
    boost::archive::xml_oarchive  ioutputxml(outputixml);  
    ioutputxml << BOOST_SERIALIZATION_NVP(outputi);  
    outputi.clear(); 
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
    cin>>rconnect_cap;
    clock_t t = clock();
    cout<<"wait..."<<endl;
    vector<float> vec(NNs,0);
    NN = vec;
    bias = vec;
    vector<vector<int>> vec1(NNs);
    W1i = vec1;
    W2i = vec1;
    a1i = vec1;
    a2i = vec1;
    vector<vector<float>> vec2(NNs);
    W1s = vec2;
    W2s = vec2;
    vec.clear();
    vec1.clear();
    vec2.clear();
    weight_start();
    afill();
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
