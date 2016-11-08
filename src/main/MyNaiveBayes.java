package main;

import weka.classifiers.AbstractClassifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class MyNaiveBayes extends AbstractClassifier {

	@Override
	public void buildClassifier(Instances instances) throws Exception {
		// check if attribute is numeric
		boolean is_numeric = false;
		for(int i=0; i<instances.numAttributes(); i++) {			
			if(instances.attribute(i).type() == 0) {
				is_numeric = true;
			}
		}

		if(is_numeric) {
			System.out.println("> Filtering dataset using NumericToNominal\n");
			NumericToNominal filter = new NumericToNominal();
			try {
				filter.setInputFormat(instances);
				Instances newData = Filter.useFilter(instances, filter);
				// System.out.println(newData.toSummaryString());
				System.out.println("Data filtered");
			} catch (Exception e) {
				System.out.println("Problem filtering instances\n");
			}
		}
	}

	/*
	 * Main program 
	 * Usage: java MyNaiveBayes <file arff>
	 */
	public static void main(String args[]) {
		
		if(args.length < 1) {
			System.out.println("Usage: java MyNaiveBayes <.arffinstancesInput>");
		}
		else {
			// Membaca instances yang diberikan  
			String arff = args[0];
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
			
			MyNaiveBayes nb = new MyNaiveBayes();
			try {
				nb.buildClassifier(instances);
			} catch (Exception e) {
				System.out.println("Problem building classifier\n");
			}
			
		}
		
		
	}
}
