#include "neuralnetwork.h"

NeuralNetwork::NeuralNetwork()
{
  srand(time(0));
  _isInitialized = false;
  _logMode = 0;
  _interact();
}

NeuralNetwork::~NeuralNetwork()
{
  if(_isInitialized)
    _memerase();
}

/*
 * PRIVATE
 */

void NeuralNetwork::_init()
{
  if(_isInitialized)
    _memerase();
  _xSample = new short [_nX * _nSamples];
  _ySample = new short [_nY * _nSamples];
  _w = new short [_nX * _nHidden];
  _v = new short [_nY * _nHidden];
  _isInitialized = true;
//  _initializeCoefficients();
}

void NeuralNetwork::_memerase()
{
  delete _xSample;
  delete _ySample;
  delete _w;
  delete _v;
}

void NeuralNetwork::_initializeCoefficients()
{
  for(int i = 0; i < _nX * _nHidden; i++)
  {
    _w[i] = rand() % 2;
  }
  for(int i = 0; i < _nY * _nHidden; i++)
  {
    _v[i] = rand() % 2;
  }
}

void NeuralNetwork::_study()
{
  short *hid = new short [_nHidden];  // hidden layer outputs
  short *y = new short [_nY];         // output layer outputs o_O
  short *xSample;                     // current sample inputs
  short *ySample;                     // current sample outputs
  short *dY = new short [_nY];        // output layer errors
  bool *cHid = new bool [_nHidden];   // need to change hidden neuron
  short hamming;                      // Hamming distance for sample's output
  short epochHamming;                 // Hamming distance for epoch
  short epochBeforeHamming;           // Hamming distance before corrections
  int *iSamples = new int [_nSamples];// sequence of indexes of samples ot study
  ofstream output("study.log", ios::out);
  _initializeCoefficients();
  for(int nEpoch = 0; nEpoch < _nEpochs; nEpoch++)
  {
    epochHamming = 0;
    epochBeforeHamming = 0;
    if(_logMode > 0)
      output << "Epoch #" << nEpoch << '\n';
    for(int i = 0; i < _nSamples; i++)
      iSamples[i] = -1;
    for(int i = 0; i < _nSamples; i++)
    {
      int index;
      while(iSamples[index = rand() % _nSamples] >= 0) {}
      iSamples[index] = i;
    }
    for(int nSample = 0; nSample < _nSamples; nSample++)
    {
      xSample = _xSample + iSamples[nSample] * _nX; // setting up x for current sample
      ySample = _ySample + iSamples[nSample] * _nY; // setting up y for current sample
      for(int i = 0; i < _nHidden; i++) // making all hidden neurons unchanged
        cHid[i] = false;
      /* count current sample */
      _countSample(xSample, y, hid);
      hamming = 0;
      for(int i = 0; i < _nY; i++)
        if(y[i] != ySample[i])
          hamming++;
      epochBeforeHamming += hamming;
      /* output info */
      if(_logMode == 2)
      {
        output << "\n Sample #" << nSample << " with own index #"
               << iSamples[nSample] << '\n' << "\n ";
        for(int i = 0; i < _nX; i++)
          output << ' ' << xSample[i];
        output << "\n ";
        for(int i = 0; i < _nY; i++)
          output << ' ' << ySample[i];
        output << "\n  Hidden neurons\n ";
        for(int i = 0; i < _nHidden; i++)
          output << ' ' << hid[i];
        output << '\n';
        output << "  Output neurons\n";
        output << ' ';
        for(int i = 0; i < _nY; i++)
          output << ' ' << y[i];
        output << "\n  Hamming before: " << setw(2) << hamming;
        output << "\n  Hidden coefficients\n     ";
        for(int i = 0; i < _nX; i++)
          output << ' ' << xSample[i];
        for(int j = 0; j < _nHidden; j++)
        {
          output << "\n  " << setw(2) << j << ':';
          for(int i = 0; i < _nX; i++)
            output << ' ' << _w[j * _nX + i];
        }
        output << "\n  Output coefficients\n     ";
        for(int i = 0; i < _nHidden; i++)
          output << ' ' << hid[i];
        for(int j = 0; j < _nY; j++)
        {
          output << "\n  " << setw(2) << j << ':';
          for(int i = 0; i < _nHidden; i++)
            output << ' ' << _v[j * _nHidden + i];
        }
        output << "\n  Studying\n";
        output << "   Output layer\n";
      }
      /* <1> check which hidden neurons should be changed */
      for(int j = 0; j < _nY; j++)
      {
        if(y[j] == 0 && ySample[j] == 1) // if we have 0 but want 1
        {
          int amount = 0; // amount of zeroes
          for(int i = 0; i < _nHidden; i++)
            if(hid[i] == 0 && cHid[i] == false) // if it is zero and unchecked
              amount++;
          if(amount == _nHidden) // if all hiddens are zeroes
          {
            int index = rand() % _nHidden;
            cHid[index] = true;
            if(_logMode == 2)
              output << "    " << setw(2) << j << ": " << index
                     << " hidden neuron should be changed\n";
          }
        }
        if(y[j] == 1 && ySample[j] == 0) // if we have 1 but want 0
        {
          if(_logMode == 2)
            output << "    " << setw(2) << j << ':';
          for(int i = 0; i < _nHidden; i++)
            if(hid[i] == 1 && cHid[i] == false)
            {
              cHid[i] = true;
              if(_logMode == 2)
                output << ' ' << i;
            }
          if(_logMode == 2)
            output << " hidden neurons should be changed\n";
        }
      }
      /* <2> changing coefficients for selected hidden neurons */
      if(_logMode == 2)
        output << "   Hidden layer\n";
      for(int j = 0; j < _nHidden; j++)
        if(cHid[j]) // if this neuron needs to be changed
        {
          if(j < _nDis) // if neuron is a disjunctor
          {
            if(hid[j] == 0) // if neuron is 0 so it needs to be true
            {
              if(_logMode == 2)
                output << " d  " << setw(2) << j << ':';
              int amount = 0; // amount of trues among inputs
              for(int i = 0; i < _nX; i++)
                if(xSample[i] == 1)
                  amount++;
              if(amount > 0)
              {
                int index = rand() % amount;
                int k = 0;
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
                if(_logMode == 2)
                  output << ' ' << index << " coef changes to true\n";
              }
              else
              {
                if(_logMode == 2)
                  output << " should be 1 but no one true input was found!\n";
              }
            }
            else // if neuron is 1 so it needs to be false
            {
              if(_logMode == 2)
                output << " d  " << setw(2) << j << ':';
              for(int i = 0; i < _nX; i++)
                if(xSample[i] == 1 && _w[j * _nX + i] == 1)
                {
                  _w[j * _nX + i] = 0;
                  if(_logMode == 2)
                    output << ' ' << i;
                }
              if(_logMode == 2)
                output << " coefs changes from 1 to 0\n";
            }
          }
          else // if neuron is a conjunctor
          {
            if(hid[j] == 0) // if neuron is 0 so it needs to be true
            {
              if(_logMode == 2)
                output << " c  " << setw(2) << j << ':';
              for(int i = 0; i < _nX; i++)
                if(xSample[i] == 0 && _w[j * _nX + i] == 0)
                {
                  _w[j * _nX + i] = 1;
                  if(_logMode == 2)
                    output << ' ' << i;
                }
              if(_logMode == 2)
                output << " coefs changes from 0 to 1\n";
            }
            else // if neuron is 1 so it needs to be false
            {
              if(_logMode == 2)
                output << " c  " << setw(2) << j << ':';
              int amount = 0; // amount of falses among inputs
              for(int i = 0; i < _nX; i++)
                if(xSample[i] == 0)
                  amount++;
              if(amount > 0)
              {
                int index = rand() % amount;
                int k = 0;
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
                if(_logMode == 2)
                  output << ' ' << index << " coef changes to false\n";
              }
              else
              {
                if(_logMode == 2)
                  output << " should be 0 but no one false input was found!\n";
              }
            }
          }
        }
      /* <3> changing coefficients for output neurons */
      _countSample(xSample, y, hid);
      if(_logMode == 2)
      {
        output << "  Output neurons\n";
        output << ' ';
        for(int i = 0; i < _nY; i++)
          output << ' ' << y[i];
        output << "\n  Hidden coefficients\n     ";
        for(int i = 0; i < _nX; i++)
          output << ' ' << xSample[i];
        for(int j = 0; j < _nHidden; j++)
        {
          output << "\n  " << setw(2) << j << ':';
          for(int i = 0; i < _nX; i++)
            output << ' ' << _w[j * _nX + i];
        }
        output << "\n  Output coefficients\n     ";
        for(int i = 0; i < _nHidden; i++)
          output << ' ' << hid[i];
        for(int j = 0; j < _nY; j++)
        {
          output << "\n  " << setw(2) << j << ':';
          for(int i = 0; i < _nHidden; i++)
            output << ' ' << _v[j * _nHidden + i];
        }
        output << "\n   Output layer\n";
      }
      for(int j = 0; j < _nY; j++)
      {
        if(y[j] == 0 && ySample[j] == 1) // if we have 0 but want 1
        {
          if(_logMode == 2)
            output << "    " << setw(2) << j << ':';
          int amount = 0; // amount of trues among hiddens
          for(int i = 0; i < _nHidden; i++)
            if(hid[i] == 1)
              amount++;
          if(amount > 0)
          {
            int index = rand() % amount;
            int k = 0;
            for(int i = 0; i < _nHidden; i++)
              if(hid[i] == 1)
              {
                if(k == index)
                {
                  _v[j * _nHidden + i] = 1;
                  break;
                }
                k++;
              }
            if(_logMode == 2)
              output << ' ' << index << " coef changes from 0 to 1\n";
          }
          else
          {
            if(_logMode == 2)
              output << " should be 1 but no one true hidden was found!\n";
          }
        }
        if(y[j] == 1 && ySample[j] == 0) // if we have 1 but want 0
        {
          if(_logMode == 2)
            output << "    " << setw(2) << j << ':';
          for(int i = 0; i < _nHidden; i++)
            if(hid[i] == 1 && _v[j * _nHidden + i] == 1)
            {
              _v[j * _nHidden + i] = 0;
              if(_logMode == 2)
                output << ' ' << i;
            }
          if(_logMode == 2)
            output << " coefs changes from 1 to 0\n";
        }
      }
      _countSample(xSample, y, hid);
      hamming = 0;
      for(int i = 0; i < _nY; i++)
        if(y[i] != ySample[i])
          hamming++;
      epochHamming += hamming;
      if(_logMode == 2)
      {
        output << "  Output neurons\n";
        output << ' ';
        for(int i = 0; i < _nY; i++)
          output << ' ' << y[i];
        output << "\n  Hamming after:  " << setw(2) << hamming << '\n';
      }
    }
    if(_logMode > 0)
    {
      output << "Before study error: " << epochBeforeHamming << '\n';
      output << "After study error: " << epochHamming << '\n';
    }
    if(epochHamming == 0 && epochBeforeHamming == 0)
    {
        output << " Yo-hou! Study finished on " << nEpoch << " epoch!\n";
      break;
    }
  }
  output.close();
  delete [] hid;
  delete [] y;
//  delete [] xSample;
//  delete [] ySample;
  delete [] dY;
  delete [] cHid;
  delete [] iSamples;
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

int NeuralNetwork::_readFromFile(string str)
{
  ifstream input(str.c_str(), ios::in);
  if(!input.is_open())
    return 0;
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
  return 1;
}

int NeuralNetwork::_readNet(string str)
{
  ifstream input(str.c_str(), ios::out);
  if(!input.is_open())
    return 0;
  input >> _nDis >> _nCon >> _nX >> _nY;
  _nHidden = _nDis + _nCon;
  _init();
  for(int j = 0; j < _nHidden; j++)\
    for(int i = 0; i < _nX; i++)
      input >> _w[j * _nX + i];

  for(int j = 0; j < _nY; j++)
    for(int i = 0; i < _nHidden; i++)
      input >> _v[j * _nHidden + i];
  input.close();
  return 1;
}

void NeuralNetwork::_writetoFile(string str)
{
  ofstream output(str.c_str(), ios::out);
  output << _nDis << '\n' << _nCon << '\n' << _nX << '\n' << _nY << '\n';
  for(int j = 0; j < _nHidden; j++)
  {
    for(int i = 0; i < _nX; i++)
      output << ' ' << _w[j * _nX + i];
    output << '\n';
  }
  for(int j = 0; j < _nY; j++)
  {
    for(int i = 0; i < _nHidden; i++)
      output << ' ' << _v[j * _nHidden + i];
    output << '\n';
  }
  output.close();
}

void NeuralNetwork::_interact()
{
  bool isExit = false;
  map <string, int> commands;
  commands["exit"] = 1;
  commands["help"] = 2;
  commands["sample"] = 3;
  commands["study"] = 4;
  commands["readnet"] = 5;
  commands["log"] = 6;
  string sCommand = "";
  cout << "Welcome!\nType \"help\" to see a list of commands.\n";
  while(!isExit)
  {
    cout << "Command: ";
    cin >> sCommand;
    switch(commands[sCommand])
    {
      case 1: // exit
        isExit = true;
        break;
      case 2: // help
        cout << "A list of commands:\n";
        cout << "exit\n";
        cout << "help\n";
        cout << "sample\n";
        cout << "study\n";
        cout << "readnet\n";
        cout << "log\n";
        break;
      case 3:
        if(_isInitialized)
        {
          char c;
          short *x = new short [_nX];
          short *hid = new short [_nHidden];
          short *y = new short [_nY];
          cout << "Enter " << _nX << " input variables:\n";
          for(int i = 0; i < _nX; i++)
          {
            cin >> c;
            if(c == 48)
              x[i] = 0;
            else
              x[i] = 1;
          }
          _countSample(x, y, hid);
          cout << "You have:";
          for(int i = 0; i < _nY; i++)
          {
            if(i % int(sqrt(_nY)) == 0)
              cout << '\n';
            cout << ' ' << y[i];
          }
          cout << '\n';
          delete [] x;
          delete [] hid;
          delete [] y;
        }
        else
        {
          cout << "Net is not initialized. Try to use \'readnet\' "
               << "or \'study\' commands.\n";
        }
        break;
      case 4:
        if(_readFromFile())
        {
          _study();
          _writetoFile();
          cout << "Net\'s study process has successfully finished!\n";
        }
        else
          cout << "File with study samples doesn\'t exist!\n";
        break;
      case 5:
        if(_readNet())
          cout << "Net parameters loaded.\n";
        else
          cout << "File with studied net doesn\'t exist!\n";
        break;
      case 6:
        {
          int temp;
          cout << "Set logging mode (0 - disable, 1 - only errors, 2 - full; "
               << "now it is " << _logMode << "): ";
          cin >> temp;
          if(temp < 0)
            temp = 0;
          if(temp > 2)
            temp = 2;
          _logMode = temp;
          cout << "Logging mode is set to " << _logMode << "!\n";
        }
        break;
      default:
        cout << "Command \'" << sCommand << "\' not found. Type \'help"
             << "\' to see a list of commands.\n";
        break;
    }
  }
}
