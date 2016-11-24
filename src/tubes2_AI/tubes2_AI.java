/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes2_AI;


//import weka.core.DenseInstance;
//import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.classifiers.Classifier;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.core.Attribute;
import weka.filters.Filter;

import java.io.Serializable;
import java.util.Set;
import java.util.Scanner;
import java.util.Random;
//import java.util.ArrayList;

import ffnn.*;
import NaiveBayesPckge.*;
import static NaiveBayesPckge.NaiveBayesMain.useFilterDiscritize;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToNominal;

/**
 *
 * @author agilajah
 */
public class tubes2_AI {
    
        static int filterr = 0;
        static String filteractive;
        static String metode;
        static String fileactive;
        static String existingmodels;
        static Instances dataSet;
        static Instances newDataSet;
        static Instances dataTest;
        static Evaluation usedEvaluation;
        static String chosenClassifier= "FFNN(default)"; //default
        static int numChosenClassifier=0; //0 FFNN, 1 NB
        static MyFFNN ffnn;
        static NaiveBayesCode NB;
        static Classifier usedClassifier;


        //for FFNN
        
        private static NominalToBinary nominalToBinaryFilter; 

        static int nInputNeuron = 0;
        static int nHiddenLayer = 1;
        static int nHiddenNeuron = 35;
        static int nOutputNeuron = 0;
        static double learningRate = 0.1;
        static int epoch = 155000;
        static double minErrorRate = 0.4;

        
        
        public static void showStatus() {
                    System.out.println();
                    System.out.println();
                    System.out.println("============================================================================================================");
                    System.out.println("[File:" + fileactive + "]" + "    " + "[Filter: " + filteractive + "]"  + "    " + "[Validation Method:" + metode + "]" + "    " +
                                        "[Model: " + existingmodels + "]" + "    " + "[Classifier: " + chosenClassifier + "]");
                    System.out.println("============================================================================================================");
                    System.out.println();
        }
        public static void saveModel(String filename) throws Exception {
            if (numChosenClassifier==0) {
                weka.core.SerializationHelper.write("/home/agilajah/Desktop/model/" + filename + ".ffnn", ffnn);
            } else {
                weka.core.SerializationHelper.write("/home/agilajah/Desktop/model/" + filename + ".nb", NB);
            }
            
        }


        public static MyFFNN loadModelFFNN(String filename) throws Exception {
                showStatus();
                //FFNN
                chosenClassifier = "FFNN";
                numChosenClassifier = 0;

                ffnn = (MyFFNN) weka.core.SerializationHelper.read("/home/agilajah/Desktop/model/" + filename + ".ffnn");

                return ffnn;
        }
        
        public static NaiveBayesCode loadModelNB(String filename) throws Exception {
                showStatus();
                //NB

                chosenClassifier = "Naive Bayes";
                numChosenClassifier = 1;
                
                NB = (NaiveBayesCode) weka.core.SerializationHelper.read("/home/agilajah/Desktop/model/" + filename + ".nb");
 
                return NB;
        }
        
        
        public static Instances useFilterDiscretize(Instances dataSet) throws Exception {
            filteractive = "Discretize";
            //set options
            String[] optionsFilter = new String[4];
            //choose the number of intervals, e.g 2:
            optionsFilter[0] = "-B";
            optionsFilter[1] = "6";
            //choose the range of attributes on which to apply the filter:
            optionsFilter[2] = "-R";
            optionsFilter[3] = "first-last";
            System.out.println("> Filtering dataset using Discretize\n");
            //Apply Discretization
            Discretize discretize = new Discretize();
            discretize.setOptions(optionsFilter);
            discretize.setInputFormat(dataSet);
            Instances newDataTemp = Filter.useFilter(dataSet, discretize);

            return newDataTemp;
        }

        public static Instances useFilterNominalToNumeric(Instances dataSet) throws Exception{
            boolean isNumeric = false;
            Instances newInstance = null;
            filteractive = "NominalToNumeric";

            // 0 = Numeric
            // 1 = Nominal
            for(int i=0; i<dataSet.numAttributes(); i++) {			
                if((dataSet.attribute(i).type() == 0) || (dataSet.attribute(i).type() == 1)) {
                    isNumeric = true;
                }
            }

            if (isNumeric) {
                System.out.println("> Filtering dataset using NumericToNominal\n");
                NumericToNominal filter = new NumericToNominal();
                try {
                        filter.setInputFormat(dataSet);
                        newInstance = Filter.useFilter(dataSet, filter);
                        //System.out.println(newInstance);
                        System.out.println("Data filtered");
                } catch (Exception e) {
                        System.out.println("Problem filtering instances\n");
                }
            }

            return newInstance;
        }
        
        public static int trainingMethodInteractive() {
  
               showStatus();
               System.out.println("Choose appropriate training method : ");
               System.out.println("1. 10 Folds Cross Validation");
               System.out.println("2. Full-Training");
               String chooseMethods = terminalInputString();
               while (!chooseMethods.equals("1") && !chooseMethods.equals("2")) {
                   chooseMethods = terminalInputString();
                   if (!chooseMethods.equals("1") && !chooseMethods.equals("2")) {
                       System.out.println("Choose correctly, please.");
                   }
               }

               if (chooseMethods.equals("1")) {
                   metode = new String("CrossValidation");
                   return 1;
               } else {
                   if (chooseMethods.equals("2")) {
                       metode = new String("Full-training");
                       return 2;
                   }
               }

               //default
               return 1;

       }


        public static Evaluation crossValidation(Instances newDataSet, Classifier c) throws  Exception {
                int seed = 1;
                int folds = 10;

                Random rand = new Random(seed);

                //create random dataset
                Instances randData = new Instances(newDataSet);
                randData.randomize(rand);

                Evaluation eval = new Evaluation(randData);
                eval.crossValidateModel(c, randData, folds, rand);

                return eval;
        }


        public static Evaluation full_training(Instances newDataSet, Classifier c) throws Exception {
                Evaluation ft = new Evaluation(newDataSet);
                ft.evaluateModel(c, newDataSet);
                return ft;
        }
        
        
        /**
        * Checks {@link Instances} data if attributes (all but the label attributes)
        * are numeric or nominal. Nominal attributes are transformed to binary by use of
        * {@link NominalToBinary} filter.
        *
        * @param dataSet instances data to be checked
        * @param inputAttributes input/feature attributes which format need to be checked
        * @return data set if it passed checks; otherwise <code>null</code>
        */
       public static Instances applyNominaltoBinaryFilter(Instances dataSet) {
               try {
                   nominalToBinaryFilter = new NominalToBinary();
                   //nominalToBinaryFilter.
                   nominalToBinaryFilter.setInputFormat(dataSet);
                   dataSet = Filter.useFilter(dataSet, nominalToBinaryFilter);
               } catch (Exception exception) {
                   nominalToBinaryFilter = null;
                   exception.printStackTrace();
               }
           

           return dataSet;
       }
        public static void modelInteractive() throws Exception {
             System.out.println("Input data test: ");
            String namafile = terminalInputString();
            System.out.println("filename: " + namafile);
                    
            try {
			DataSource source = new DataSource("dataSet/" + namafile + ".arff");
			dataTest = source.getDataSet();
			if(dataTest.classIndex() == -1) {
				dataTest.setClassIndex(dataTest.numAttributes() - 1);
			}
			System.out.println("Loaded instances: " + namafile + ".arff\n");
		} catch (Exception  e) {
			System.out.println("Problem loading instances: " + namafile + "\n");
		}
            // check whether user have model or not
            showStatus();
            System.out.println("Wanna input your existing model? (Yes/No)");
            String modelChoose = terminalInputString();
            String tempChoose = modelChoose.toLowerCase();
            System.out.println();
            System.out.println("What is your classifier type?");
            System.out.println("1. FFNN");
            System.out.println("2. Naive Bayes");
            String classifierType = terminalInputString();
            if (classifierType.equals("1")) {
                    numChosenClassifier = 0;
                    chosenClassifier = "FFNN";
            } else {
                numChosenClassifier = 1;
                chosenClassifier = "Naive Bayes";
                
            }
            if (tempChoose.equals("yes") || tempChoose.equals("y")) {
                System.out.println();
                System.out.println("----------------------------------------------------------------");
                System.out.println("|                       Model Loader                            ");
                System.out.println("----------------------------------------------------------------");
                System.out.println();
                
                System.out.print("Input your model filename : ");
                String modelFileName = terminalInputString();
                
                
                if (classifierType.equals("1")) {
                    usedClassifier = (MyFFNN) loadModelFFNN(modelFileName);
                    loadData();
                    newDataSet = ffnn.normalize(newDataSet);
                    
                } else {
                    usedClassifier = (NaiveBayesCode) loadModelNB(modelFileName);
                    loadData();
                    newDataSet = useFilterDiscretize(newDataSet);
                }
                existingmodels = new String(modelFileName);
                System.out.println("Model succesfully loaded.");
                //filterInteractive();
                System.out.println("Dataset loaded.");

                if (trainingMethodInteractive() == 1) {
                    usedEvaluation = crossValidation(dataTest, usedClassifier);
                } else
                    usedEvaluation = full_training(dataTest, usedClassifier);

            } else { // if user has no existing model
                System.out.println();

                System.out.println("Dataset loaded.");

                //building classifier model
                if (numChosenClassifier == 0) {
                    chosenClassifier = "FFNN";
                    usedClassifier = (MyFFNN) baseClassifier(dataSet);
                } else {
                    chosenClassifier = "Naive Bayes";
                    usedClassifier = (NaiveBayesCode) baseClassifier(dataSet);
                }
                
                System.out.println("----------------------------------------------------------------");
                System.out.println("|                       Model Builder                           ");
                System.out.println("----------------------------------------------------------------");
                System.out.println();
                System.out.println("Model successfully built.");
                System.out.println();
                System.out.println();
                showStatus();
                System.out.println("Want to save model? (Yes/No)");

                String chooseSaveModel = terminalInputString();
                tempChoose = chooseSaveModel.toLowerCase();
                if (tempChoose.equals("yes") || tempChoose.equals("y")) {
                    System.out.println();
                    System.out.print("Input your filename here: ");
                    String filename = terminalInputString();
                    saveModel(filename);
                    System.out.println();
                    existingmodels = filename;
                    System.out.println("Model saved");
                } else {
                    System.out.println();
                    System.out.println("Model not saved");
                }
                //asking user whether using filter or nah
                //filterInteractive();
                
                if (trainingMethodInteractive() == 1) {
                    if (numChosenClassifier==0) {

                       //filtering the datatest
                       dataTest = applyNominaltoBinaryFilter(dataTest);
                       dataTest=ffnn.normalize(dataTest); 
                    } else {
                        dataTest=useFilterDiscretize(dataTest);
                    }
                    
                    usedEvaluation = crossValidation(dataTest, usedClassifier);
                } else {
                    if (numChosenClassifier==0) {
                       dataTest=ffnn.normalize(dataTest); 
                    } else {
                        dataTest=useFilterDiscretize(dataTest);
                    }
                    usedEvaluation = full_training(dataTest, usedClassifier);
                }
                    
                    

            }


        }//end of modelinteractive
            
        /*
            Choosing baseClassifier before training
        */
        public static Classifier baseClassifier(Instances dataSetTemp) throws Exception {
                    
                    showStatus();
                    
                                  
                    if (numChosenClassifier==0) {
                        //aktivasi FFNN by default
                        nInputNeuron = dataSetTemp.numAttributes()-1;
                        nOutputNeuron = dataSetTemp.numClasses();
                        loadData();
                        ffnn = new MyFFNN(nInputNeuron, nHiddenLayer, nHiddenNeuron, nOutputNeuron, learningRate, epoch, minErrorRate);
                        chosenClassifier = "FFNN"; //by default
                        //applying filter
                        newDataSet = applyNominaltoBinaryFilter(newDataSet);
                        newDataSet = ffnn.normalize(newDataSet);
                        return ffnn;
                        
                    } else {
                        //aktivasi naive bayes
                        numChosenClassifier = 1;
                        chosenClassifier = "Naive Bayes";
                        loadData();
                        NB = new NaiveBayesCode(newDataSet.numAttributes());
                        try {
                            //            newDataSet = useFilterNominalToNumeric(newDataSet);
                            newDataSet = useFilterDiscritize(newDataSet);
                        } catch (Exception e) {
                                    System.out.println("Problem when use filter : " + e);
                        }
                        
                        
                        return NB;
                    }
                    
                    
        }
   
        
        /*
            a method that used for integer input
        */
        public static int terminalInputInteger() {
                //create the scanner
                Scanner terminalInput = new Scanner(System.in);
                int result = terminalInput.nextInt();
                
                return result;
        }
        
        /*
            a method that used for input string
        */
        public static String terminalInputString() {
                //create the scanner
                Scanner terminalInput = new Scanner(System.in);
                String result = terminalInput.next();
                
                return result;
        }
        
        /*
            copying dataset to newdataset that used in a whole process
        */
        public static void loadData() throws Exception {
                newDataSet = dataSet;
        }
        
        
        /*
            a method to classify new instance
        */
        public static void addNewInstance() throws Exception {
            /*//create the scanner
            Scanner terminalInput = new Scanner(System.in);
            System.out.println();
            System.out.println("====================================");
            System.out.println("[File:" + fileactive + "]" + "[Filter: " + filteractive + "]" + "[Method:" + metode + "]");
            System.out.println("[Model: " + existingmodels + "]");
            System.out.println("====================================");
            System.out.println("Write down your new instance : ");

            ArrayList<Attribute> atts = new ArrayList<Attribute>();
            ArrayList<String> classVal = new ArrayList<String>();
            classVal.add("Iris-setosa");
            classVal.add("Iris-versicolor");
            classVal.add("Iris-virginica");

            atts.add(new Attribute("sepallength"));
            atts.add(new Attribute("sepalwidth"));
            atts.add(new Attribute("petallength"));
            atts.add(new Attribute("petalwidth"));

            atts.add(new Attribute("@@class@@",classVal));

            double[] attValues = new double[dataSet.numAttributes()];
            for (int i=0; i < dataSet.numAttributes()-1;i++) {
                attValues[i] = terminalInput.nextDouble();
            }
            Discretize discretize = new Discretize();
            String s = terminalInput.nextLine();

            Instance instance1 = new DenseInstance(1.0, attValues);


            instance1.setDataset(newDataSet);
            if (filterr == 1) {
                discretize.setInputFormat(dataSet);
                discretize.input(instance1);
            }
            System.out.println(instance1);
            int classify1 = (int) usedClassifier.classifyInstance(instance1);
            System.out.print("Prediction Class : ");
            System.out.println(classVal.get(classify1)); */
        }
        
        public static void main(String args[]) throws Exception{
                //status
                showStatus();
                System.out.println("Input dataset file name: ");
                //read filename
                String namafile = terminalInputString();
                fileactive = new String(namafile);
                
		try {
			DataSource source = new DataSource("dataSet/" + namafile + ".arff");
			dataSet = source.getDataSet();
			if(dataSet.classIndex() == -1) {
				dataSet.setClassIndex(dataSet.numAttributes() - 1);
			}
			System.out.println("Loaded instances: " + namafile + ".arff\n");
		} catch (Exception  e) {
			System.out.println("Problem loading instances: " + namafile + "\n");
		}
                
                modelInteractive();
                
                System.out.println(usedClassifier.toString());
		try {
			usedClassifier.buildClassifier(newDataSet);
			
			Evaluation eval = new Evaluation(dataTest);			
			eval.evaluateModel(usedClassifier, dataTest);
			showStatus();
			System.out.println(eval.toSummaryString());		// Summary of Training
			System.out.println(eval.toMatrixString());
			
			System.out.println("Error rate: "+eval.errorRate()*100+" %"); // Printing Training Mean root squared error
			System.out.println("Accuracy: "+(1-eval.errorRate())*100+" %");
                        if (numChosenClassifier==0) {
                            System.out.println("Epoch: "+ ffnn.getEpoch());
                        }
			
		} catch (Exception e) {
			e.printStackTrace();
		}	
                
                /*
                System.out.println("Wanna test with new instance? (Yes/No)");
                String yesno = terminalInputString();
                String tempyesno = yesno.toLowerCase();
                if (tempyesno.equals("yes")) {
                        addNewInstance();
                }*/
                


        }
}
