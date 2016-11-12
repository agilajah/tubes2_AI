package ffnn;

import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class MyFFNN extends AbstractClassifier {
	
	// Attributes
	Neuron inputLayer[];			// input layer
	Neuron hiddenLayers[][];		// hidden layers
	Neuron outputLayer[];			// output layer
	
	double learningRate;		// learning rate
	
	// Constructor
	MyFFNN(int nInputNeuron, int nHiddenLayer, int nHiddenNeuron, int nOutputNeuron, double learningRate) {
		// Initialize input layer
		inputLayer = new Neuron[nInputNeuron+1]; 			// + 1 for bias neuron
		inputLayer[inputLayer.length-1] = new Neuron(1);	// bias neuron
		
		for(int i=0; i<inputLayer.length-1; i++) {
			inputLayer[i] = new Neuron();
		}
		
		// Initialize hidden layers
		hiddenLayers = new Neuron[nHiddenLayer][nHiddenNeuron+1];		// + 1 for bias neuron
		
		for(int i=0; i<hiddenLayers.length; i++) {
			hiddenLayers[i][hiddenLayers[i].length-1] = new Neuron(1);		// bias neuron
			for(int j=0; j<hiddenLayers[i].length-1; j++) {
				hiddenLayers[i][j] = new Neuron();
			}
		}
		
		// Initialize output layer
		outputLayer = new Neuron[nOutputNeuron];
		
		for(int i=0; i<outputLayer.length; i++) {
			outputLayer[i] = new Neuron();
		}
		
		// Create connections between neurons
		Neuron[][] layers = new Neuron[1+nHiddenLayer+1][];
		int count = 0;
		layers[count++] = inputLayer;		
		for(int j = 0; j<nHiddenLayer; j++) {
			layers[count++] = hiddenLayers[j];
		}
		layers[count++] = outputLayer;
				
		for(int c=0; c<count-2; c++) {
			for(int i=0; i<layers[c].length; i++) {
				for(int j=0; j<layers[c+1].length-1; j++) {
					Connection con = new Connection(layers[c][i], layers[c+1][j]);
					layers[c][i].addConnection(con);
					layers[c+1][j].addConnection(con);
				}
			}
		}
		
		// for output layer
		for(int i=0; i<layers[count-2].length; i++) {
			for(int j=0; j<layers[count-1].length; j++) {
				Connection con = new Connection(layers[count-2][i], layers[count-1][j]);
				layers[count-2][i].addConnection(con);
				layers[count-1][j].addConnection(con);
			}
		}
	}
	
	
	
	// Feed an instance to the network
	public double[] feedForward(double[] instance) {
		// Feed the input layer
		for(int i=0; i<inputLayer.length-1; i++) {
			inputLayer[i].setOutput(instance[i]);
		}
		
		// Feed the hidden layers
		for(int i=0; i<hiddenLayers.length; i++) {
			for(int j=0; j<hiddenLayers[i].length-1; j++) {
				hiddenLayers[i][j].calculateOutput();
			}
		}
		
		// Feed the output layer
		double[] result = new double[outputLayer.length];
		for(int i=0; i<outputLayer.length; i++) {
			outputLayer[i].calculateOutput();
			result[i] = outputLayer[i].getOutput();
		}
		
		return result;
	}
	
//	public double train(double[] inputs, double answer) {
//		double result = feedForward(inputs);
//		
//		double deltaOutput = result*(1-result) * (answer-result);
//		
//		ArrayList connections = outputLayer[0].getConnections();
//        for (int i = 0; i < connections.size(); i++) {
//            Connection c = (Connection) connections.get(i);
//            Neuron neuron = c.getFrom();
//            double output = neuron.getOutput();
//            double deltaWeight = output*deltaOutput;
//            c.adjustWeight(learningRate*deltaWeight);
//        }
//        
//        // ADJUST HIDDEN WEIGHTS
//        for(int i=0; i<hiddenLayers.length; i++) {
//        	 for (int j = 0; j < hiddenLayers[i].length; j++) {
//                 connections = hiddenLayers[i][j].getConnections();
//                 double sum  = 0;
//                 // Sum output delta * hidden layer connections (just one output)
//                 for (int k = 0; k < connections.size(); k++) {
//                     Connection c = (Connection) connections.get(k);
//                     // Is this a connection from hidden layer to next layer (output)?
//                     if (c.getFrom() == hiddenLayers[i][j]) {
//                         sum += c.getWeight()*deltaOutput;
//                     }
//                 }    
//                 // Then adjust the weights coming in based:
//                 // Above sum * derivative of sigmoid output function for hidden neurons
//                 for (int k = 0; k < connections.size(); k++) {
//                     Connection c = (Connection) connections.get(k);
//                     // Is this a connection from previous layer (input) to hidden layer?
//                     if (c.getTo() == hiddenLayers[i][j]) {
//                         double output = hiddenLayers[i][j].getOutput();
//                         double deltaHidden = output * (1 - output);  // Derivative of sigmoid(x)
//                         deltaHidden *= sum;   // Would sum for all outputs if more than one output
//                         Neuron neuron = c.getFrom();
//                         double deltaWeight = neuron.getOutput()*deltaHidden;
//                         c.adjustWeight(learningRate*deltaWeight);
//                     }
//                 } 
//             }
//        }
//       
//        return result;
//	}
	
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {

	    // since all the output values are needed.
	    // They are calculated manually here and the values collected.
	    int numClasses = instance.numClasses();
	    double[] out = new double[numClasses];
	    
	    int class_index = instance.classIndex();
		int num_attributes = instance.numAttributes();
		double inputs[] = new double[num_attributes];
		
		for(int i=0, j=0; i<num_attributes; i++) {
			if(i != class_index) {
				inputs[j++] = instance.value(i);
			}
		}
		
	    out = this.feedForward(inputs);
	    
	    if (instance.classAttribute().type() == Attribute.NUMERIC) {
	    	return out;
	    }

	    // now normalize the array
	    double count = 0;
	    for (int i = 0; i < numClasses; i++) {
	    	count += out[i];
	    }
	    
	    for (int i = 0; i < numClasses; i++) {
	      out[i] /= count;
	    }
	    
	    return out;
	  }
	
	public static void main(String args[]) {
		String arff = "ffnn/iris.arff";
		System.out.println("> Loading instances: " + arff + "\n");
		Instances instances = null;
		try {
			DataSource source = new DataSource(arff);
			instances = source.getDataSet();
			if(instances.classIndex() == -1) {
				instances.setClassIndex(instances.numAttributes() - 1);
			}				
			System.out.println("Loaded instances: " + arff + "\n");
			// System.out.println(instances.toSummaryString());
		} catch (Exception e) {
			System.out.println("Problem loading instances: " + arff+ "\n");
		}
				
		int nInputNeuron = instances.numAttributes()-1;
		int nHiddenLayer = 1;
		int nHiddenNeuron = 2;
		int nOutputNeuron = instances.numClasses();
		double learningRate = 0.01;
		
		MyFFNN ffnn = new MyFFNN(nInputNeuron, nHiddenLayer, nHiddenNeuron, nOutputNeuron, learningRate);
		try {
			ffnn.buildClassifier(instances);
			Evaluation eval = new Evaluation(instances);
			eval.evaluateModel(ffnn, instances);
			eval.crossValidateModel(ffnn, instances, 10, new Random(1));
			System.out.println(eval.errorRate()); // Printing Training Mean root squared error
			System.out.println(eval.toSummaryString());		// Summary of Training
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}



	@Override
	public void buildClassifier(Instances instances) throws Exception {
		Instance inst = instances.instance(0);	// get first instance
		int class_index = instances.classIndex();
		int num_attributes = inst.numAttributes();
		double inputs[] = new double[num_attributes];
		
		for(int i=0, j=0; i<num_attributes; i++) {
			if(i != class_index) {
				inputs[j++] = inst.value(i);
			}
		}		
				
	}
	
}
