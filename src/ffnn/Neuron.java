package ffnn;

import java.io.Serializable;
import java.util.ArrayList;

public class Neuron implements Serializable {
	
	double output;				// output value
	ArrayList connections;		// neuron's connections
	
	Neuron() {
		output = 0;
		connections = new ArrayList();
	}
	
	Neuron(int biasWeight) {
		output = biasWeight;
		connections = new ArrayList();
	}
	
	void addConnection(Connection c) {
		connections.add(c);
	}

	// Calculate output of neuron
	public void calculateOutput() {
		double sum = 0;
		for(int i=0; i< connections.size(); i++) {
			Connection con = (Connection) connections.get(i);
			Neuron from = con.getFrom();
			Neuron to = con.getTo();
			if(to == this) {
				sum += from.getOutput() * con.getWeight();
			}
		}
		output = sigmoid(sum);
	}
	
	// Sigmoid function
    public double sigmoid(double x) {
        return 1.0f / (1.0f + (double) Math.exp(-x));
    }
	
	public double getOutput() {
		return this.output;			
	}
	
	public void setOutput(double output) {
		this.output = output;			
	}

	public ArrayList getConnections() {
		return connections;
	}
}
