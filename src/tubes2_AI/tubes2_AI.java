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
//import weka.core.Attribute;
import java.io.Serializable;

import java.util.Scanner;
import java.util.Random;
//import java.util.ArrayList;

import ffnn.*;
import NaiveBayesPckge.*;

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
        static Evaluation usedEvaluation;
        static String chosenClassifier= "FFNN(default)"; //default
        static int numChosenClassifier=0; //0 FFNN, 1 NB
        static MyFFNN ffnn;
        static NaiveBayesCode NB;
        static Classifier usedClassifier;


        //for FFNN

        static int nInputNeuron = 0;
        static int nHiddenLayer = 1;
        static int nHiddenNeuron = 15;
        static int nOutputNeuron = 0;
        static double learningRate = 0.3;
        static int epoch = 10000;
        static double minErrorRate = 0.10;


        public static void saveModel(String filename) throws Exception {
            if (numChosenClassifier==0) {
                weka.core.SerializationHelper.write("/home/agilajah/Desktop/model/" + filename + ".model", ffnn);
            } else {
                weka.core.SerializationHelper.write("/home/agilajah/Desktop/model/" + filename + ".model", NB);
            }
            
        }


        public static MyFFNN loadModelFFNN(String filename) throws Exception {
                System.out.println("[File:" + fileactive + "]" + "[Filter: " + filteractive + "]" + "[Method:" + metode + "]");
                System.out.println("[Model: " + existingmodels + "]" + "[Classifier: " + chosenClassifier + "]");
                //FFNN
                chosenClassifier = "FFNN";
                numChosenClassifier = 0;

                ffnn = (MyFFNN) weka.core.SerializationHelper.read("/home/agilajah/Desktop/model/" + filename + ".model");

                return ffnn;
        }
        
        public static NaiveBayesCode loadModelNB(String filename) throws Exception {
                System.out.println("[File:" + fileactive + "]" + "[Filter: " + filteractive + "]" + "[Method:" + metode + "]");
                System.out.println("[Model: " + existingmodels + "]" + "[Classifier: " + chosenClassifier + "]");

                //NB

                chosenClassifier = "Naive Bayes";
                numChosenClassifier = 1;

                return NB;
        }
        
        public static int trainingMethodInteractive() {
               //create the scanner
               Scanner terminalInput = new Scanner(System.in);

               System.out.println();
               System.out.println("====================================");
               System.out.println("[File:" + fileactive + "]" + "[Filter: " + filteractive + "]" + "[Method:" + metode + "]");
               System.out.println("[Model: " + existingmodels + "]" + "[Classifier: " + chosenClassifier + "]");
               System.out.println("====================================");
               System.out.println("Choose appropriate training method : ");
               System.out.println("1. 10 Folds Cross Validation");
               System.out.println("2. Full-Training");
               String chooseMethods = terminalInput.nextLine();
               while (!chooseMethods.equals("1") && !chooseMethods.equals("2")) {
                   chooseMethods = terminalInput.nextLine();
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

        public static void modelInteractive() throws Exception {

            // check whether user have model or not
            System.out.println();
            System.out.println("====================================");
            System.out.println("[File:" + fileactive + "]" + "[Filter: " + filteractive + "]" + "[Method:" + metode + "]");
            System.out.println("[Model: " + existingmodels + "]" + "[Classifier: " + chosenClassifier + "]");
            System.out.println("====================================");
            System.out.println("Wanna input your existing model? (Yes/No)");
            String modelChoose = terminalInputString();
            String tempChoose = modelChoose.toLowerCase();
            System.out.println("What is your classifier type?");
            System.out.println("1. FFNN");
            System.out.println("2. Naive Bayes");
            int classifierType = terminalInputInteger();
            if (tempChoose.equals("yes") || tempChoose.equals("y")) {
                System.out.println();
                System.out.println("----------------------------------------------------------------");
                System.out.println("|                       Model Loader                            ");
                System.out.println("----------------------------------------------------------------");
                System.out.println();
                System.out.println();
                System.out.println("[File:" + fileactive + "]" + "[Filter: " + filteractive + "]" + "[Method:" + metode + "]");
                System.out.println("[Model: " + existingmodels + "]" + "[Classifier: " + chosenClassifier + "]");

                System.out.print("Input your model filename : ");
                String modelFileName = terminalInputString();
                
                
                if (classifierType == 1) {
                    numChosenClassifier = 0;
                    usedClassifier = (MyFFNN) loadModelFFNN(modelFileName);
                    loadData();
                    newDataSet = ffnn.normalize(newDataSet);
                    
                } else {
                    numChosenClassifier = 1;
                    usedClassifier = (NaiveBayesCode) loadModelNB(modelFileName);
                    loadData();
                }
                existingmodels = new String(modelFileName);
                System.out.println("Model succesfully loaded.");
                //loadData();
                //filterInteractive();
                System.out.println("Dataset loaded.");

                if (trainingMethodInteractive() == 1) {
                    usedEvaluation = crossValidation(newDataSet, usedClassifier);
                } else
                    usedEvaluation = full_training(newDataSet, usedClassifier);

            } else { // if user has no existing model
                System.out.println();

                System.out.println("Dataset loaded.");

                //building classifier model
                if (numChosenClassifier == 0) {
                    usedClassifier = (MyFFNN) baseClassifier(dataSet);
                } else {
                    usedClassifier = (NaiveBayesCode) baseClassifier(dataSet);
                }
                
                System.out.println("----------------------------------------------------------------");
                System.out.println("|                       Model Builder                           ");
                System.out.println("----------------------------------------------------------------");
                System.out.println();
                System.out.println("Model successfully built.");
                System.out.println();
                System.out.println();
                System.out.println("====================================");
                System.out.println("[File:" + fileactive + "]" + "[Filter: " + filteractive + "]" + "[Method:" + metode + "]");
                System.out.println("[Model: " + existingmodels + "]" + "[Classifier: " + chosenClassifier + "]");
                System.out.println("====================================");
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
                    usedEvaluation = crossValidation(newDataSet, usedClassifier);
                } else
                    usedEvaluation = full_training(newDataSet, usedClassifier);

            }


        }//end of modelinteractive
            
        /*
            Choosing baseClassifier before training
        */
        public static Classifier baseClassifier(Instances dataSetTemp) throws Exception {
                    
                    System.out.println();
                    System.out.println("====================================");
                    System.out.println("[File:" + fileactive + "]" + "[Filter: " + filteractive + "]" + "[Method:" + metode + "]");
                    System.out.println("[Model: " + existingmodels + "]" + "[Classifier: " + chosenClassifier + "]");
                    System.out.println("====================================");
                    System.out.println();
                    
                                  
                    if (numChosenClassifier==0) {
                        //aktivasi FFNN by default
                        nInputNeuron = dataSetTemp.numAttributes()-1;
                        nOutputNeuron = dataSetTemp.numClasses();
                        ffnn = new MyFFNN(nInputNeuron, nHiddenLayer, nHiddenNeuron, nOutputNeuron, learningRate, epoch, minErrorRate);
                        chosenClassifier = "FFNN"; //by default
                        loadData();
                        newDataSet = ffnn.normalize(newDataSet);
                        return ffnn;
                        
                    } else {
                        //aktivasi naive bayes
                        numChosenClassifier = 1;
                        chosenClassifier = "Naive Bayes";
                        loadData();
                        
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
                System.out.println("[File:" + fileactive + "]" + "[Filter: " + filteractive + "]" + "[Method:" + metode + "]");
                System.out.println("[Model: " + existingmodels + "]" + "[Classifier: " + chosenClassifier + "]");
                System.out.println("Input dataset file name: ");
                //read filename
                String namafile = terminalInputString();
                fileactive = new String(namafile);
                
		try {
			DataSource source = new DataSource("dataSet/" + namafile + ".arff");
			dataSet = source.getDataSet();
			if(dataSet.classIndex() == -1) {
				dataSet.setClassIndex(dataSet.numAttributes() - 1);
//				instances.setClassIndex(0);
			}
			System.out.println("Loaded instances: " + namafile + ".arff\n");
		} catch (Exception  e) {
			System.out.println("Problem loading instances: " + namafile + "\n");
		}
                
                modelInteractive();
                
                System.out.println(usedClassifier.toString());
		try {
			usedClassifier.buildClassifier(newDataSet);
			
			Evaluation eval = new Evaluation(newDataSet);			
			eval.evaluateModel(usedClassifier, newDataSet);
			
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