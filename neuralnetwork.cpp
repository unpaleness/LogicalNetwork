#include "neuralnetwork.h"

NeuralNetwork::NeuralNetwork()
{
  srand(time(0));
  _isInitialized = false;
  _readFromFile();
  _study();
  _writetoFile();
}

NeuralNetwork::~NeuralNetwork()
{
  if(_isInitialized)
  {
    delete _xSample;
    delete _ySample;
    delete _w;
    delete _v;
  }
}

/*
 * PRIVATE
 */

void NeuralNetwork::_init()
{
  if(!_isInitialized)
  {
    _xSample = new short [_nX * _nSamples];
    _ySample = new short [_nY * _nSamples];
    _w = new short [_nX * _nHidden];
    _v = new short [_nY * _nHidden];
    _isInitialized = true;
  }
  for(int i = 0; i < _nX * _nHidden; i++)
    _w[i] = rand() % 1;
  for(int i = 0; i < _nY * _nHidden; i++)
    _v[i] = rand() % 1;
}

void NeuralNetwork::_study()
{
  short *hid = new short [_nHidden]; // hidden layer outputs
  short *y = new short [_nY]; // output layer outputs o_O
  short *xSample;
  short *ySample;
  short *dHid = new short [_nHidden]; // hidden layer errors
  short *dY = new short [_nY]; // output layer errors
  ofstream output("study.log", ios::out);
  for(int nEpoch = 0; nEpoch < _nEpochs; nEpoch++)
  {
    output << "\nEpoch #" << nEpoch << '\n';
    for(int nSample = 0; nSample < _nSamples; nSample++)
    {
      xSample = _xSample + nSample * _nX;
      ySample = _ySample + nSample * _nY;
      /* count current sample */
      _countSample(xSample, y, hid);
      /* count errors */
      // 1) output layer errors
      for(int i = 0; i < _nY; i++)
        dY[i] = ySample[i] - y[i];
      // 2) hidden layer errors
//      for(int j = 0; j < _nHidden; j++)
//      {
//        dHid[j] = 0;
//        for(int i = 0; i < _nY; i++)
//          dHid[j] += dY[i] * _v[i * _nHidden + j];
//        if(dHid[j] > 1)
//          dHid[j] = 1;
//      }
      /* recount coefficients */
      // 1) before hidden layer
//      for(int j = 0; j < _nHidden; j++)
//        for(int i = 0; i < _nX; i++)
//        {
//          _w[j * _nX + i] -= dHid[j] * xSample[i];
//          _w[j * _nX + i] = abs(_w[j * _nX + i]);
//        }
      // 2) before output layer
//      for(int j = 0; j < _nY; j++)
//        for(int i = 0; i < _nHidden; i++)
//        {
//          _v[j * _nHidden + i] -= dY[j] * hid[i];
//          _v[j * _nHidden + i] = abs(_v[j * _nHidden + i]);
//        }
      output << "\n Sample #" << nSample << '\n';
      output << ' ';
      for(int i = 0; i < _nX; i++)
        output << ' ' << xSample[i];
      output << "\n ";
      for(int i = 0; i < _nY; i++)
        output << ' ' << ySample[i];
      output << "\n ";
      for(int i = 0; i < _nY; i++)
        output << ' ' << y[i];
      output << "\n  Errors hidden:";
      for(int i = 0; i < _nHidden; i++)
        output << ' ' << dHid[i];
      output << "\n  Neurons hidden";
      for(int j = 0; j < _nHidden; j++)
      {
        output << "\n  " << setw(2) << j << ':';
        for(int i = 0; i < _nX; i++)
          output << ' ' << _w[i * _nHidden + j];
      }
      output << "\n  Errors output:";
      for(int i = 0; i < _nY; i++)
        output << ' ' << dY[i];
      output << "\n  Neurons output";
      for(int j = 0; j < _nY; j++)
      {
        output << "\n  " << setw(2) << j << ':';
        for(int i = 0; i < _nHidden; i++)
          output << ' ' << _v[i * _nY + j];
      }
      output << '\n';
    }
  }
  output.close();
  delete [] hid;
  delete [] y;
//  delete [] xSample;
//  delete [] ySample;
  delete [] dY;
  delete [] dHid;
}

void NeuralNetwork::_countSample(short *x, short *y, short *hid)
{
  short temp;
  // hidden layer
  for(int j = 0; j < _nDis; j++) // for each disjunctor
  {
    hid[j] = 0;
    for(int i = 0; i < _nX; i++) // for each input variable
      hid[j] += _w[j * _nX + i] * x[i];
    if(hid[j] > 1)
      hid[j] = 1;
//    cout << ' ' << hid[j];
  }
  for(int j = _nDis; j < _nHidden; j++) // for each conjunctor
  {
    hid[j] = 1;
    for(int i = 0; i < _nX; i++) // for each input variable
    {
      temp = _w[j * _nX + i] + x[i];
      if(temp > 1)
        temp = 1;
      hid[j] *= temp;
    }
//    cout << ' ' << hid[j];
  }
//  cout << '\n';
  // output layer
  for(int j = 0; j < _nY; j++) // all disjunctors
  {
    y[j] = 0;
//    cout << "y[" << j << "]=";
    for(int i = 0; i < _nHidden; i++)
    {
      y[j] += _v[j * _nHidden + i] * hid[i];
//      cout << '+' << _v[i * _nY + j] << '*' << hid[i];
    }
    if(y[j] > 1)
      y[j] = 1;
//    cout << '=' << y[j] << '\n';
  }
//  cout << '\n';
}

void NeuralNetwork::_readFromFile(string str)
{
  ifstream input(str.c_str(), ios::in);
  input >> _nDis >> _nCon >> _nX >> _nY >> _nEpochs >> _nSamples;
  _nHidden = _nDis + _nCon;
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
