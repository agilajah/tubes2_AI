package ffnn;

import java.util.ArrayList;
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
	int counterEpoch = 0;
	int epoch;
	double minErrorRate;
	
	// Constructor
	MyFFNN(int nInputNeuron, int nHiddenLayer, int nHiddenNeuron, int nOutputNeuron, double learningRate, int epoch, double minErrorRate) {
		this.learningRate = learningRate;
		this.epoch = epoch;
		this.minErrorRate = minErrorRate;
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
	
	public int getEpoch() {
		return this.epoch;
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
	
	public double[] train(double[] instance, double d) throws Exception {
		// Hitung output dari instance
		double[] result = feedForward(instance);
		
		// Tentukan output maksimal 
		int iMax = 0;
		double max = 0;
		for(int i=0; i < result.length; i++) {
			if(result[i] > max) {
				iMax = i;
				max = result[i];
			}
		}
		
		boolean guessTrue = false;
		if(iMax == (int) d) {
			guessTrue = true;
//			System.out.println("sama");
		}
		
		// Buat array baru untuk answer
		double[] answers = new double[result.length];
		for(int i=0; i < answers.length; i++) {
			if(i == (int)d) {
				answers[i] = 1;
			}
			else {
				answers[i] = 0;
			}
		}
		
		// BACKPROPOGATION
        // Apply Delta to connections between hidden and output
		double[] deltaOutput = new double[result.length];
		for(int i=0; i < outputLayer.length; i++) {
			// This is where the error correction all starts
	        // Derivative of sigmoid output function * diff between known and guess
			if(guessTrue) {
				deltaOutput[i] = 0;
			}
			else {
				deltaOutput[i] = result[i]*(1-result[i]) * (answers[i]-result[i]);	
//				System.out.print(deltaOutput[i]+" *** ");
			}
//			System.out.println(deltaOutput[i]);
			outputLayer[i].setDeltaOutput(deltaOutput[i]);
			
			ArrayList connections = outputLayer[i].getConnections();
	        for (int j = 0; j < connections.size(); j++) {
	            Connection c = (Connection) connections.get(j);
	            Neuron neuron = c.getFrom();
	            double output = neuron.getOutput();
	            double deltaWeight = output*deltaOutput[i];
	            c.adjustWeight(learningRate*deltaWeight);
	        }
		}		
        
        // ADJUST HIDDEN WEIGHTS
        for(int i=hiddenLayers.length-1; i>=0; i--) {
        	 for (int j = 0; j < hiddenLayers[i].length; j++) {
        		 ArrayList connections = hiddenLayers[i][j].getConnections();
                 double sum  = 0;
                 // Sum output delta * hidden layer connections 
                 for (int k = 0; k < connections.size(); k++) {
                     Connection c = (Connection) connections.get(k);
                     // Is this a connection from hidden layer to next layer (output)?
                     if (c.getFrom() == hiddenLayers[i][j]) {
                         sum += c.getWeight()*c.getTo().getDeltaOutput();
                     }
                 }    
                 // Then adjust the weights coming in based:
                 // Above sum * derivative of sigmoid output function for hidden neurons
                 for (int k = 0; k < connections.size(); k++) {
                     Connection c = (Connection) connections.get(k);
                     // Is this a connection from previous layer (input) to hidden layer?
                     if (c.getTo() == hiddenLayers[i][j]) {
                         double output = hiddenLayers[i][j].getOutput();
                         double deltaHidden = output * (1 - output);  // Derivative of sigmoid(x)
                         deltaHidden *= sum;   // Would sum for all outputs if more than one output
                         Neuron neuron = c.getFrom();
                         double deltaWeight = neuron.getOutput()*deltaHidden;
                         c.adjustWeight(learningRate*deltaWeight);
                     }
                 } 
             }
        }
       
        return result;
	}
	
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
	
	@Override
	public void buildClassifier(Instances instances) throws Exception {
		int class_index = instances.classIndex();
		int num_attributes = instances.numAttributes();
				
		for(int i=0; i<instances.size(); i++) {
			double inputs[] = new double[num_attributes];
			
			for(int k=0, j=0; k<num_attributes; k++) {
				if(k != class_index) {
					inputs[j++] = instances.get(i).value(k);
//					System.out.print(instances.get(i).value(k)+" ");
				}
			}	
			
			double[] result = this.train(inputs, instances.get(i).classValue());

		}
		counterEpoch++;
		Evaluation eval = new Evaluation(instances);
		eval.evaluateModel(this, instances);
		if(eval.errorRate() > this.minErrorRate && counterEpoch < this.epoch) {
			this.buildClassifier(instances);
		}
	}
	
	
	public static void main(String args[]) {
		// Read arff file
		String arff = "main/iris.arff";
		System.out.println("> Loading instances: " + arff + "\n");
		Instances instances = null;
		try {
			DataSource source = new DataSource(arff);
			instances = source.getDataSet();
			if(instances.classIndex() == -1) {
				instances.setClassIndex(instances.numAttributes() - 1);
			}				
			System.out.println("Loaded instances: " + arff + "\n");
		} catch (Exception e) {
			System.out.println("Problem loading instances: " + arff+ "\n");
		}
				
		
		int nInputNeuron = instances.numAttributes()-1;
		int nHiddenLayer = 1;
		int nHiddenNeuron = 4;
		int nOutputNeuron = instances.numClasses();
		double learningRate = 0.25;
		int epoch = 10000;
		double minErrorRate = 0.01;
		
		MyFFNN ffnn = new MyFFNN(nInputNeuron, nHiddenLayer, nHiddenNeuron, nOutputNeuron, learningRate, epoch, minErrorRate);
		try {
			ffnn.buildClassifier(instances);
			Evaluation eval = new Evaluation(instances);
			
			eval.evaluateModel(ffnn, instances);
//			eval.crossValidateModel(ffnn, instances, 10, new Random(1));
			
			System.out.println(eval.toSummaryString());		// Summary of Training
			System.out.println(eval.toMatrixString());
			
			System.out.println("Error rate: "+eval.errorRate()); // Printing Training Mean root squared error
			System.out.println("Epoch: "+ffnn.getEpoch());
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
