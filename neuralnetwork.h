#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <ctime>
#include <cstdlib>
#include <cmath>

using namespace std;

class NeuralNetwork
{
public:
  NeuralNetwork();
  ~NeuralNetwork();

private:
  bool _isInitialized;
  int _nDis;
  int _nCon;
  int _nHidden;
  int _nX;
  int _nY;
  int _nEpochs;
  int _nSamples;
  short *_xSample;
  short *_ySample;
  short *_w;
  short *_v;

  void _init(); // initializes network
  void _initializeCoefficients();
  void _study(); // studies network
  void _countHidden(short *x, short *hid); // counts hidden layer
  void _countOutput(short *hid, short *y); // counts output layer
  void _countSample(short *x, short *y, short *hid); // counts a sample
  void _readFromFile(string str = "input.txt");
  void _writetoFile(string str = "output.txt");
};

#endif // NEURALNETWORK_H
