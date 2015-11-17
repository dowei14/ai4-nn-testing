#include <utils/ann-framework/ann.h>
#include <utils/ann-framework/backpropagation.h>
#include <utils/ann-framework/neuron.h>
#include <iostream>
#include <fstream>
#include <random>
#include <sstream>
#include <string>

using namespace std;

class CSMTL : public ANN
{
	public:
		CSMTL();
		enum { nn_inputs = 15, nn_hidden = 20, nn_outputs=2 };
		
};

CSMTL::CSMTL() {

	//set transfer function
//	setDefaultTransferFunction(ANN::tanhFunction());
	setDefaultTransferFunction(ANN::logisticFunction());

	setNeuronNumber(nn_inputs+nn_hidden+nn_outputs); // total number of neurons

	// create random number generator between -1 and 1
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator
    std::uniform_real_distribution<> distr(-1, 1); // define the range

	// initialize weights
	for (int h=nn_inputs; h<(nn_inputs+nn_hidden);h++){
		for (int i=0; i<nn_inputs;i++){
			w(h,i,distr(eng));
		}
	}
	for (int o=nn_inputs+nn_hidden; o<(nn_inputs+nn_hidden+nn_outputs);o++){
		for (int h=nn_inputs; h<(nn_inputs+nn_hidden);h++){
			w(o,h,distr(eng));
		}
	}
}

int main(int argc, char **argv) {
	CSMTL ann;

	// create a topolocial sorting of our network. This is required for the
	// backpropagation algorithm as "Full Batch mode"
	ann.updateTopologicalSort();

	// create backpropagation object
	Backpropagation trainer;
	trainer.setNeuralNetwork(&ann);
	for (int i=0; i<ann.nn_inputs;i++){
		trainer.defineInputNeuron(i, ann.getNeuron(i));
	}
	for (int o=ann.nn_inputs+ann.nn_hidden; o<(ann.nn_inputs+ann.nn_hidden+ann.nn_outputs);o++){
		trainer.defineOutputNeuron(o, ann.getNeuron(o));
	}	
	trainer.includeAllSynapses();
	trainer.includeAllNeuronBiases();
	trainer.setLearningRate(0.01);
	
	
	ifstream data_inputs("../data/inCSMTL1.txt"); //opening an input stream for file test.txt
	ifstream data_outputs("../data/outCSMTL1.txt"); //opening an input stream for file test.txt
	std::string line_out;
	for(std::string line_in; std::getline(data_inputs, line_in); )   //read stream line by line
	{
		// create Training pattern
		TrainingPattern* p = new TrainingPattern;
		
		// add inputs
		std::istringstream in(line_in);      //make a stream for the line itself
		//cout<<"Inputs: ";
		for (int i=0;i<ann.nn_inputs;i++){
			double x;
			in >> x;                  //and read the first whitespace-separated token
			p->inputs[i] = x;
			//cout<<x<<" ";
		}

		// add outputs
		std::getline(data_outputs, line_out);
		std::istringstream out(line_out);
		//cout<<" Outputs: ";
		for (int o=0;o<ann.nn_outputs;o++){
			double x;
			out >> x;                  //and read the first whitespace-separated token
			p->outputs[o] = x;
			//cout<<x<<" ";
		}
		// add pattern to trainer
		trainer.addTrainingPattern(p);
		//cout<<endl;
	}
	trainer.learn(5);
	std::cout << ann.dumpWeights();


/*
  // create a topolocial sorting of our network. This is required for the
  // backpropagation algorithm as "Full Batch mode"
  ann.updateTopologicalSort();

  // training data
  const double data[4][3] = {{0, 0, 0},
                             {0, 1, 1},
                             {1, 0, 1},
                             {1, 1, 0}}; // {input 1, input 2, output}

  // create backpropagation object
  Backpropagation trainer;
  trainer.setNeuralNetwork(&ann);
  trainer.defineInputNeuron(0, ann.getNeuron(0));
  trainer.defineInputNeuron(1, ann.getNeuron(1));
  trainer.defineOutputNeuron(0, ann.getNeuron(4));
  trainer.includeAllSynapses();
  trainer.includeAllNeuronBiases();
  trainer.setLearningRate(0.1);
  for (int i=0; i<4; i++)
  {
    TrainingPattern* p = new TrainingPattern;
    p->inputs[0]  = data[i][0]; // text file here from your own created inputs
    p->inputs[1]  = data[i][1];
    p->outputs[0] = data[i][2];
    trainer.addTrainingPattern(p);
  }

  trainer.learn(50000);

  ann.setInput(0,0);
  ann.setInput(1,0);
  ann.feedForwardStep();
  std::cout << "0 0 => " << ann.getOutput(4) << std::endl;

  ann.setInput(0,1);
  ann.setInput(1,0);
  ann.feedForwardStep();
  std::cout << "1 0 => " << ann.getOutput(4) << std::endl;

  ann.setInput(0,0);
  ann.setInput(1,1);
  ann.feedForwardStep();
  std::cout << "0 1 => " << ann.getOutput(4) << std::endl;

  ann.setInput(0,1);
  ann.setInput(1,1);
  ann.feedForwardStep();
  std::cout << "1 1 => " << ann.getOutput(4) << std::endl;

  std::cout << ann.dumpWeights();
  */
}

