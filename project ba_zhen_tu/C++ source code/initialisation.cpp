/*
Copyright (c) 2020 2win9s
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

another reason is so that I can have all of these variables in a list and group them here so their purpose can be explained

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
//to add some more randomness into the connecting of neurons (1 - random gaussian) * connect_base is used mean of gaussian is 0
float rconnect_sdeviation;                              //the standard deviation for the random number (for connections) 
float rconnect_cap;                                     //the cap on the absolute value of the random number (we don't want it to suddenly jump out of control)


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


//prints 1d vector
template<typename r>            
void vecprint1d(const r& vec){
    cout << "{";
    if (!vec.empty()) for (int x = 0; x < vec.size(); x++) {
        cout<<vec[x]<<" ,";
    } 
    cout<<"}"<<endl;
}

//prints a 2d vector
template<typename s>            
void vecprint2d(const s& vec){
    cout << "{" << endl;
    if (!vec.empty()) for (int x = 0; x < vec.size(); x++) {
        cout << "{";
        if (!vec[x].empty()) {
            for (int y = 0; y < vec[x].size() - 1; y++) {
                cout << vec[x][y] << ", ";
            }
            cout << vec[x][vec[x].size() - 1];
        }
        cout << "}" << endl;
    }
    cout << "}" << endl;
}

//prints the size of all the vectors in a 2d vector
template<typename s>  
void vec2dsize(const s& vec) {          
    cout<<"main vector size "<<vec.size()<<endl;
    cout << "{";
    if (!vec.empty()) for (int x = 0; x < vec.size(); x++) {
            cout<<"{"<<vec[x].size()<<"}";
    }
    cout << "}" <<endl;
}

// function initialises list of available connects for each neuron (a1i and a2i), must be called before syncinit()
void afill(){ 
    for(int x = 0;x < a1i.size();x++){
        for(int y = 0; y < x - 1; y++){
            a1i[x].insert(a1i[x].end(),y);
        }
    }
    for(int x = 1; x < a2i.size();x++){
        for(int y = x + 1; y < NNs; y++){
            a2i[x].insert(a2i[x].end(),y);
        }
    }
    for(int i = 1; i < NNs - 1;i++){
        a2i[0].insert(a2i[0].end(),i);
    }
}

//function for staring the weights, this function makes sure that each neuron is connected to the one behind it, the first neuron is connected to the last;
//this ensures that information will flow through all the neurons
void weight_start(){
    normal_distribution<float> distribution(0,1);
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
    int tas = (Lthreadz * 3);
    #pragma omp parallel num_threads(Lthreadz) proc_bind(spread)
    {
        #pragma omp for simd nowait
        for(int x = 0;x < NNs - 1; x++){
            normal_distribution<double> d(0,rconnect_sdeviation);
            int a = W1i[x].size();
            double mis = (x - a);
            a += W2i[x].size();
            double randnm;
            long double k;
            randnm = d(twisting);
            if(abs(randnm) > rconnect_cap){
                randnm = rconnect_cap;
                if(rand() % 2 == 0){
                    randnm *= -1;
                }
            }
            k = 1 - randnm;
            double connectn = 1 *  k * mis * connect_base;
            int connectnm;
            if(connectn > 1){
            connectnm = floor(connectn);
            }
            else{
                uniform_real_distribution<double> rk(0.0,1.0);
                double chance = rk(twisting);
                if(connectn >= chance){
                    connectnm = 1;
                }
                else{
                    connectnm = 0;
                }
            }
            if(connectnm >= a1i[x].size()){
                connectnm = a1i[x].size();
            }
            int lmt = a1i[x].size();
            float zz = W1i[x].size() + connectnm;
            long double xy = 2.0 / zz;
            double He = sqrt(xy);

            normal_distribution<float> F(0,He); 

            for(int i = 0; i < connectnm; i++){

                int ng = twisting() % lmt;
                W1i[x].insert(W1i[x].end(),a1i[x][ng]);

                float in = F(twisting); 
                W1s[x].insert(W1s[x].end(),in);

                a1i[x].erase(a1i[x].begin() + ng);

                --lmt;
            }
        }
    #pragma omp for simd
    for(int y = 1;y < NNs ;y++){
        normal_distribution<double> dis(0,rconnect_sdeviation);
        int a = W2i[y].size();
        double mis = NNs - (y + 1);
        double randnm;
        long double k;
        a += W1i[y].size();
        randnm = dis(twisting);
        if(abs(randnm) > rconnect_cap){
            randnm = rconnect_cap;
            if(twisting() % 2 == 0){
                randnm *= -1;
            }
        }
        k = 1 - randnm;
        double connectn = 1 * k * mis * connect_base;
        int connectnm;
        if(connectn > 1){
           connectnm = floor(connectn);
        }
        else{
            uniform_real_distribution<double> tri(0.0,1.0);
            double chance = tri(twisting);
            if(connectn >= chance){
                connectnm = 1;
            }
            else{
                connectnm = 0;
            }
        }
        if(connectnm >= a2i[y].size()){
            connectnm = a2i[y].size();
        }
        int lmt = a2i[y].size();
        float zz = W2i[y].size() + connectnm;
        long double xy = 2.0 / zz;
        double He = sqrt(xy);

        normal_distribution<float> al(0,He);

        for(int i = 0; i < connectnm; i++){

            int ng = twisting() % lmt;
            W2i[y].insert(W2i[y].end(),a2i[y][ng]);

            float in = al(twisting); 
            W2s[y].insert(W2s[y].end(),in);

            a2i[y].erase(a2i[y].begin() + ng);

            --lmt;
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
    cout<<"enter input neuron indices"<<endl<<"must start with lowest index"<<endl<<"e.g. if your list is {1,2} wait for prompt type 1 wait for next prompt then type 2 ..."<<endl;
    for(int x = 0; x < length; x++){
        cout<<"enter input neuron index (remember vector indexing starts from 0)"<<endl;
        cin>>ind;
        inputi.insert(inputi.end(),ind);
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
    cout<<"enter input neuron indices"<<endl<<"must start with lowest index"<<endl<<"e.g. if your list is {1,2} wait for prompt type 1 wait for next prompt then type 2 ..."<<endl;
    for(int x = 0; x < length; x++){
        cout<<"enter output neuron index (remember vector indexing starts from 0)"<<endl;
        cin>>ind;
        outputi.insert(outputi.end(),ind);
    }
}


//creates the binary files
void savebinf(){
    ofstream ntwkbin("NN.bin",ofstream::trunc);
    boost::archive::binary_oarchive  ntbin(ntwkbin);
    ntbin << NN;
    ofstream W1ibin("W1i.bin",ofstream::trunc); 
    boost::archive::binary_oarchive  iW1bin(W1ibin); 
    iW1bin << NN; 
    ofstream a1ibin("a1i.bin",ofstream::trunc); 
    boost::archive::binary_oarchive  ia1bin(a1ibin); 
    ia1bin << a1i; 
    ofstream W2ibin("W2i.bin",ofstream::trunc); 
    boost::archive::binary_oarchive  iW2bin(W2ibin); 
    iW2bin << W2i; 
    ofstream a2ibin("a2i.bin",ofstream::trunc); 
    boost::archive::binary_oarchive  ia2bin(a2ibin); 
    ia2bin << a2i; 
    ofstream W1sbin("W1s.bin",ofstream::trunc); 
    boost::archive::binary_oarchive  sW1bin(W1sbin); 
    sW1bin << W1s; 
    ofstream W2sbin("W2s.bin",ofstream::trunc); 
    boost::archive::binary_oarchive  sW2bin(W2sbin); 
    sW2bin << W2s;
    ofstream biasbin("bias.bin",ofstream::trunc); 
    boost::archive::binary_oarchive  biasesbin(biasbin); 
    biasesbin << bias;
    ofstream inputibin("inputi.bin",ofstream::trunc); 
    boost::archive::binary_oarchive  iinputbin(inputibin); 
    iinputbin << inputi;  
    ofstream outputibin("outputi.bin",ofstream::trunc); 
    boost::archive::binary_oarchive  ioutputbin(outputibin); 
    ioutputbin << outputi;  
}


//creates the xml files
void savexmlf(){ 
    ofstream ntwkxml("NN.xml",ofstream::trunc); 
    boost::archive::xml_oarchive  ntxml(ntwkxml); 
    ntxml << BOOST_SERIALIZATION_NVP(NN); 
    ofstream W1ixml("W1i.xml",ofstream::trunc);  
    boost::archive::xml_oarchive  iW1xml(W1ixml);  
    iW1xml << BOOST_SERIALIZATION_NVP(W1i);  
    ofstream a1ixml("a1i.xml",ofstream::trunc);  
    boost::archive::xml_oarchive  ia1xml(a1ixml);  
    ia1xml << BOOST_SERIALIZATION_NVP(a1i);  
    ofstream W2ixml("W2i.xml",ofstream::trunc);  
    boost::archive::xml_oarchive  iW2xml(W2ixml);  
    iW2xml << BOOST_SERIALIZATION_NVP(W2i);  
    ofstream a2ixml("a2i.xml",ofstream::trunc);  
    boost::archive::xml_oarchive  ia2xml(a2ixml);  
    ia2xml << BOOST_SERIALIZATION_NVP(a2i);  
    ofstream W1sxml("W1s.xml",ofstream::trunc);  
    boost::archive::xml_oarchive  sW1xml(W1sxml);  
    sW1xml << BOOST_SERIALIZATION_NVP(W1s);  
    ofstream W2sxml("W2s.xml",ofstream::trunc);  
    boost::archive::xml_oarchive  sW2xml(W2sxml);  
    sW2xml << BOOST_SERIALIZATION_NVP(W2s);  
    ofstream biasxml("bias.xml",ofstream::trunc);  
    boost::archive::xml_oarchive  biasesxml(biasxml);  
    biasesxml << BOOST_SERIALIZATION_NVP(bias);   
    ofstream inputixml("inputi.xml",ofstream::trunc);  
    boost::archive::xml_oarchive  iinputxml(inputixml);  
    iinputxml << BOOST_SERIALIZATION_NVP(inputi); 
    ofstream outputixml("outputi.xml",ofstream::trunc);  
    boost::archive::xml_oarchive  ioutputxml(outputixml);  
    ioutputxml << BOOST_SERIALIZATION_NVP(outputi);   
}


int main(){
    string ttt;
    omp_set_dynamic(0);
    vector<int> i = {};
    vector<float> fl = {};
    cout<<"when entering numbers use only decimal fractions and decimal integers, no fractions, only 1 decimal point is allowed per number"<<endl;
    cout<<"enter number of logical processors/number of threads you are allowed concurrently"<<endl;
    cin>>Lthreadz;
    notnum(Lthreadz);
    omp_set_num_threads(Lthreadz);
    cout<<"enter number of neurons"<<endl;
    cin>>NNs;
    notnum(NNs);
    vector<float> vec(NNs,0);
    NN = vec;
    bias = vec;
    vector<vector<int>> vec1(NNs,i);
    W1i = vec1;
    W2i = vec1;
    a1i = vec1;
    a2i = vec1;
    vector<vector<float>> vec2(NNs,fl);
    W1s = vec2;
    W2s = vec2;
    i.clear();
    fl.clear();
    vec.clear();
    vec1.clear();
    vec2.clear();
    inputscan();
    outputscan();
    cout<<"enter connect base for initialisation"<<endl;
    cin>>connect_base;
    cout<<"enter rconnectrate mean for initialisation"<<endl;
    cin>>rconnect_mean;
    cout<<"enter rconnectrate standard deviation"<<endl;
    cin>>rconnect_sdeviation;
    cout<<"enter rconnectrate cap "<<endl;
    cin>>rconnect_cap;
    weight_start();
    afill();
    syncinit();
    while(true){
        cout<<"type xml for xml file, bin for binary files or both for both xml and binary files"<<endl;
        cin>>ttt;
        if(ttt == "xml"){
            savexmlf();
            break;
        }
        else if(ttt == "bin"){
            savebinf();
            break;
        }
        else if(ttt == "both"){
            savexmlf();
            savebinf();
            break;
        }
        else{
            cout<<"error: invalid input; enter xml for xml file, bin for binary files (omit the .), or both for copies of both file types"<<endl;
        }
    }
    cout<<"files have been created, you can now move on"<<endl;
    return 0;
}
