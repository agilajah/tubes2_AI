//Daniel Shiffman
//The Nature of Code, Fall 2006
//Neural Network

// Input Neuron Class
// Has additional functionality to receive beginning input

package ffnn.example;

public class InputNeuron extends Neuron {
    public InputNeuron() {
        super();
    }
    
    public InputNeuron(int i) {
        super(i);
    }

    public void input(double d) {
        output = d;
    }

}
