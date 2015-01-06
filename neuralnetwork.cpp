#include "neuralnetwork.h"

NeuralNetwork::NeuralNetwork()
{
  _isInitialized = false;
  _readFromFile();
  _writetoFile();
}

NeuralNetwork::~NeuralNetwork()
{
  if(_isInitialized)
  {
    delete _xSample;
    delete _ySample;
    delete _v;
    delete _w;
  }
}

/*
 * PRIVATE
 */

void NeuralNetwork::_init()
{
  if(!_isInitialized)
  {
    _xSample = new short [ _nX * _nSamples ];
    _ySample = new short [ _nY * _nSamples ];
    _w = new short [ _nX * (_nDis + _nCon) ];
    _v = new short [ _nY * (_nDis + _nCon) ];
    _isInitialized = true;
  }
}

void NeuralNetwork::_readFromFile(string str)
{
  ifstream input(str.c_str(), ios::in);
  input >> _nDis >> _nCon >> _nX >> _nY >> _nEpochs >> _nSamples;
  _init();
  for(int j = 0; j < _nSamples; j++)
  {
    for(int i = 0; i < _nX; i++)
      input >> _xSample[j * _nX + i];
    for(int i = 0; i < _nY; i++)
      input >> _ySample[j * _nY + i];
  }
  input.close();
}

void NeuralNetwork::_writetoFile(string str)
{
  ofstream output(str.c_str(), ios::out);
  output << "Disjunctors: " <<  _nDis << '\n';
  output << "Conjunctors: " << _nCon << '\n';
  output << "Inputs: " << _nX << '\n';
  output << "Outputs: " << _nY << '\n';
  output << "Epochs: " << _nEpochs << '\n';
  output << "Samples for study: " << _nSamples << '\n';
  for(int j = 0; j < _nSamples; j++)
  {
    output << j + 1 << " sample:\n";
    for(int i = 0; i < _nX; i++)
      output << ' ' << _xSample[j * _nX + i];
    output << '\n';
    for(int i = 0; i < _nY; i++)
      output << ' ' << _ySample[j * _nY + i];
    output << '\n';
  }
  output.close();
}
