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
    _w[i] = rand() % 2;
  for(int i = 0; i < _nY * _nHidden; i++)
    _v[i] = rand() % 2;
}

void NeuralNetwork::_study()
{
  short *hid = new short [_nHidden];  // hidden layer outputs
  short *y = new short [_nY];         // output layer outputs o_O
  short *xSample;                     // current sample inputs
  short *ySample;                     // current sample outputs
  short *dY = new short [_nY];        // output layer errors
  bool *cHid = new bool [_nHidden];   // need to change hidden neuron
  ofstream output("study.log", ios::out);
  for(int nEpoch = 0; nEpoch < _nEpochs; nEpoch++)
  {
    output << "\nEpoch #" << nEpoch << '\n';
    for(int nSample = 0; nSample < _nSamples; nSample++)
    {
      xSample = _xSample + nSample * _nX;
      ySample = _ySample + nSample * _nY;
      for(int i = 0; i < _nHidden; i++)
        cHid[i] = false;
      /* count current sample */
      _countSample(xSample, y, hid);
      /* output info */
      output << "\n Sample #" << nSample << '\n';
      output << ' ';
      for(int i = 0; i < _nX; i++)
        output << ' ' << xSample[i];
      output << "\n ";
      for(int i = 0; i < _nY; i++)
        output << ' ' << ySample[i];
      output << "\n ";
      for(int i = 0; i < _nHidden; i++)
        output << ' ' << hid[i];
      output << "\n ";
      for(int i = 0; i < _nY; i++)
        output << ' ' << y[i];
      output << "\n  Neurons hidden\n     ";
      for(int i = 0; i < _nX; i++)
        output << ' ' << xSample[i];
      for(int j = 0; j < _nHidden; j++)
      {
        output << "\n  " << setw(2) << j << ':';
        for(int i = 0; i < _nX; i++)
          output << ' ' << _w[i * _nHidden + j];
      }
      output << "\n  Neurons output\n     ";
      for(int i = 0; i < _nHidden; i++)
        output << ' ' << hid[i];
      for(int j = 0; j < _nY; j++)
      {
        output << "\n  " << setw(2) << j << ':';
        for(int i = 0; i < _nHidden; i++)
          output << ' ' << _v[i * _nY + j];
      }
      output << '\n';
      output << "  Studying\n";
      /* <1> check which hidden neurons should be changed */
      output << "   Output neurons\n";
      for(int j = 0; j < _nY; j++)
      {
        if(y[j] == 1 && ySample[j] == 0) // we have 1 and expect 0
                                         // so we need every unit to be 0
        {
          output << "   " << setw(2) << j << ':';
          for(int i = 0; i < _nHidden; i++)
            if(hid[i] == 1 && _v[j * _nHidden + i] == 1)
            {
              cHid[i] = true;
              output << ' ' << i;
            }
          output << " hidden neurons should be changed\n";
        }
        if(y[j] == 0 && ySample[j] == 1) // we have 0 and expect 1 so we
                                         // need at least one unit to be 1
        {
          output << "   " << setw(2) << j << ':';
          int amount = 0;
          int index;
          int k = 0;
          for(int i = 0; i < _nHidden; i++)
            if(hid[i] == 0 && _v[j * _nHidden + i] == 1)
              amount++;
          if(amount > 0)
          {
            index = rand() % amount;
            for(int i = 0; i < _nHidden; i++)
              if(hid[i] == 0 && _v[j * _nHidden + i] == 1)
              {
                if(k == index)
                {
                  cHid[i] = true;
                  output << ' ' << i;
                  break;
                }
                k++;
              }
          }
          else
          {
            for(int i = 0; i < _nHidden; i++)
              if(hid[i] == 0)
                amount++;
            index = rand() % amount;
            for(int i = 0; i < _nHidden; i++)
              if(hid[i] == 0)
              {
                if(k == index)
                {
                  cHid[i] = true;
                  output << ' ' << i;
                  break;
                }
                k++;
              }
          }
          output << " hidden neurons should be changed\n";
        }
      }
      /* <2> changing coefficients for selected hidden neurons */
      output << "   Hidden neurons\n";
      /* Disjunctors */
      for(int j = 0; j < _nDis; j++)
      {
        if(cHid[j]) // if it needs to be changed
        {
          output << "   " << setw(2) << j << ':';
          if(hid[j] == 1) // and now it is 1
          {
            for(int i = 0; i < _nX; i++)
              if(xSample[i] == 1)
              {
                _w[j * _nX + i] = 0;
                output << ' ' << i;
              }
            output << " coefs should be 0\n";
          }
          else // and now it is 0
          {
            int amount = 0;
            int index;
            int k = 0;
            for(int i = 0; i < _nX; i++)
              if(xSample[i] == 1)
                amount++;
            if(amount > 0)
            {
              index = rand() % amount;
              for(int i = 0; i < _nX; i++)
                if(xSample[i] == 1)
                {
                  if(k == index)
                  {
                    _w[j * _nX + i] = 1;
                    break;
                  }
                  k++;
                }
              output << ' ' << index << " coef should be 1";
            }
            output << '\n';
          }
        }
      }
      /* Conjunctors */
      for(int j = _nDis; j < _nHidden; j++)
        if(hid[j]) // if it needs to be changed
        {
          output << "   " << setw(2) << j << ':';
          if(hid[j] == 0) // and now it is 0
          {
            for(int i = 0; i < _nX; i++)
              if(xSample[i] == 0)
              {
                _w[j * _nX + i] = 1;
                output << ' ' << i;
              }
            output << " coefs should be 1\n";
          }
          else // and now it is 1
          {
            int amount = 0;
            int index;
            int k = 0;
            for(int i = 0; i < _nX; i++)
              if(xSample[i] == 0)
                amount++;
            if(amount > 0)
            {
              index = rand() % amount;
              for(int i = 0; i < _nX; i++)
                if(xSample[i] == 0)
                {
                  if(k == index)
                  {
                    _w[j * _nX + i] = 0;
                    break;
                  }
                  k++;
                }
              output << ' ' << index << " coef should be 0";
            }
            output << '\n';
          }
        }
      /* <3> changing coefficients for output neurons */
      output << "   Output neurons\n";
      _countSample(xSample, y, hid);
      for(int j = 0; j < _nY; j++)
      {
        if(y[j] == 1 && ySample[j] == 0) // we have 1 and expect 0
                                         // so we need every unit to be 0
        {
          output << "   " << setw(2) << j << ':';
          for(int i = 0; i < _nHidden; i++)
            if(hid[j] == 1)
            {
              _v[j * _nHidden + i] = 0;
              output << ' ' << i;
            }
          output << " coefs should be 0\n";
        }
        if(y[j] == 0 && ySample[j] == 1) // we have 0 and expect 1
                                        // so we need at least one unit to be 1
        {
          output << "   " << setw(2) << j << ':';
          int amount = 0;
          int index;
          int k = 0;
          for(int i = 0; i < _nHidden; i++)
            if(hid[j] == 1)
              amount++;
          if(amount > 0)
          {
            index = rand() % amount;
            for(int i = 0; i < _nHidden; i++)
              if(hid[j] == 1)
              {
                if(k == index)
                {
                  _v[j * _nHidden + i] = 1;
                  break;
                }
                k++;
              }
            output << ' ' << index << " coef should be 1";
          }
          output << '\n';
        }
      }
      output << "\n  ";
      for(int i = 0; i < _nY; i++)
        output << ' ' << y[i];
      output << "\n";
    }
  }
  output.close();
  delete [] hid;
  delete [] y;
//  delete [] xSample;
//  delete [] ySample;
  delete [] dY;
  delete [] cHid;
}

void NeuralNetwork::_countHidden(short *x, short *hid)
{
  short temp;
  for(int j = 0; j < _nDis; j++) // for each disjunctor
  {
    hid[j] = 0;
    for(int i = 0; i < _nX; i++) // for each input variable
      hid[j] += _w[j * _nX + i] * x[i];
    if(hid[j] > 1)
      hid[j] = 1;
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
  }
}

void NeuralNetwork::_countOutput(short *hid, short *y)
{
  // output layer
  for(int j = 0; j < _nY; j++) // all disjunctors
  {
    y[j] = 0;
    for(int i = 0; i < _nHidden; i++)
      y[j] += _v[j * _nHidden + i] * hid[i];
    if(y[j] > 1)
      y[j] = 1;
  }
}

void NeuralNetwork::_countSample(short *x, short *y, short *hid)
{
  _countHidden(x, hid);
  _countOutput(hid, y);
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
