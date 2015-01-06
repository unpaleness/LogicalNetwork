#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <fstream>
#include <string>

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
  int _nX;
  int _nY;
  int _nEpochs;
  int _nSamples;
  short *_xSample;
  short *_ySample;
  short *_w;
  short *_v;

  void _init(); // initializes network
  void _readFromFile(string str = "input.txt");
  void _writetoFile(string str = "output.txt");
};

#endif // NEURALNETWORK_H
