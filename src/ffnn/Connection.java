package ffnn;

import java.io.Serializable;

public class Connection implements Serializable {
	Neuron from;	
	Neuron to;
	double weight;
	
	Connection(Neuron from, Neuron to) {
		this.from = from;
		this.to = to;
		this.weight = (double) Math.random()*2-1;	// random weight
	}

	public Neuron getFrom() {
		return this.from;
	}

	public Neuron getTo() {
		return this.to;
	}

	public double getWeight() {
		return this.weight;
	}

	public void adjustWeight(double deltaWeight) {
		this.weight += deltaWeight;		
	}
}
