package ffnn;

import java.io.FileReader;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;

public class TestMultilayerPerceptron {

	public static void main(String[] args) {
		
		try {
			String filepath = "src/ffnn/iris.arff";
			FileReader trainReader = new FileReader(filepath);
			Instances train = new Instances(trainReader);
			train.setClassIndex(train.numAttributes() - 1);
			// Instance of NN
			MultilayerPerceptron mlp = new MultilayerPerceptron();
			// Setting parameters
			mlp.setLearningRate(0.1);
			mlp.setMomentum(0.2);
			mlp.setTrainingTime(2000);
			mlp.setHiddenLayers("3");
			mlp.buildClassifier(train);
			
			Evaluation eval = new Evaluation(train);
			eval.evaluateModel(mlp, train);
			eval.crossValidateModel(mlp, train, 10, new Random(1));
			System.out.println(eval.errorRate()); // Printing Training Mean root squared error
			System.out.println(eval.toSummaryString());		// Summary of Training
			
			
		}
		catch (Exception e) {
			e.printStackTrace();
		}

		
	}

}
