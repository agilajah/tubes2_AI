/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package NaiveBayesPckge;

import java.text.DecimalFormat;
import java.util.Random;
import java.util.Scanner;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToNominal;


/**
 *
 * @author Ghifari
 */
public class NaiveBayesMain {    
    static J48 tree;
        
    public static Instances useFilterDiscritize(Instances dataSet) throws Exception {
        //set options
        String[] optionsFilter = new String[4];
        //choose the number of intervals, e.g 2:
        optionsFilter[0] = "-B";
        optionsFilter[1] = "5";
        //choose the range of attributes on which to apply the filter:
        optionsFilter[2] = "-R";
        optionsFilter[3] = "first-last";

        //Apply Discretization
        Discretize discretize = new Discretize();
        discretize.setOptions(optionsFilter);
        discretize.setInputFormat(dataSet);
        Instances newDataTemp = Filter.useFilter(dataSet, discretize);
        return newDataTemp;
    }
    
    public static Instances useFilterNominalToNumeric(Instances instance) throws Exception{
        boolean isNumeric = false;
        Instances newInstance = null;
        
        // 0 = Numeric
        // 1 = Nominal
        for(int i=0; i<instance.numAttributes(); i++) {			
            if((instance.attribute(i).type() == 0) || (instance.attribute(i).type() == 1)) {
                isNumeric = true;
            }
        }
        
        if (isNumeric) {
            System.out.println("> Filtering dataset using NumericToNominal\n");
            NumericToNominal filter = new NumericToNominal();
            try {
                    filter.setInputFormat(instance);
                    newInstance = Filter.useFilter(instance, filter);
                    //System.out.println(newInstance);
                    System.out.println("Data filtered");
            } catch (Exception e) {
                    System.out.println("Problem filtering instances\n");
            }
        }
        
        return newInstance;
    }
    
    public static void saveModel(NaiveBayesCode naives, String filename) throws Exception {
        weka.core.SerializationHelper.write("D:/ITB/semester 5/AI/tugas/tubes2/savemodel" + filename + ".nbayes", naives);
    }


    public static Classifier loadModel(String filename) throws Exception {
        Classifier loader = (J48) weka.core.SerializationHelper.read("/home/agilajah/IdeaProjects/tucil2_AI/out/" + filename + ".model");

        return loader;
    }    
    
    public static void main(String[] args) {
        Instances instance;
        DecimalFormat df = new DecimalFormat("#.####");
        Scanner scan = new Scanner(System.in);
        InputNaiveBayes nb = new InputNaiveBayes();
        NaiveBayesCode naive;
        double minErrorRate = 0.2;
        double errorRates;
        double accuracy;
        String inputWeka;
        String outputWeka;
        boolean done = false;
        
        System.out.println("   Welcome to the Naive Bayes System");
        System.out.println("========================================");
        System.out.println("");
        
        System.out.print("Now input arff path file : ");
//        inputWeka = scan.nextLine();
        inputWeka = "zdata/mush.arff";
        instance = nb.readFileUseWeka(inputWeka);
        
        try {
//            instance = useFilterNominalToNumeric(instance);
            instance = useFilterDiscritize(instance);
        } catch (Exception e) {
            System.out.println("Problem when use filter : " + e);
        }
        
        
        naive = new NaiveBayesCode(instance.numAttributes(), minErrorRate);
        try {
            naive.buildClassifier(instance);
            
            Evaluation eval = new Evaluation(instance);
            eval.evaluateModel(naive, instance);
            
            System.out.println(eval.toSummaryString());		// Summary of Training
            System.out.println(eval.toMatrixString());
            
            errorRates = eval.incorrect() / eval.numInstances() * 100;
            accuracy = eval.correct() / eval.numInstances() * 100;
            
            System.out.println("Accuracy: " + df.format(accuracy) + " %");
            System.out.println("Error rate: " + df.format(errorRates) + " %"); // Printing Training Mean root squared error
            done = true;
        } catch (Exception e) {
            System.out.println("Problem : " + e);
        }
        
        
        
//        printCoba(instance);
    }
    
    public static void firstQuestion() {
        String choice;
        Scanner scan = new Scanner(System.in);
        
        System.out.println("Do you want to load the mode or do you want to test the data ?");
        System.out.println("   1. Load model");
        System.out.println("   2. Test the data");
        System.out.print("Which one do you want : ");
        choice = scan.nextLine();
        
        while (!choice.equals("1") || !choice.equals("2")) {
            System.out.println("   1. Load model");
            System.out.println("   2. Test the data");
            System.out.print("Type the number of your decision please : ");
            choice = scan.nextLine();
        }
        
        
        if (choice.equals("1")) {
            
        }
    }
    
    public static void lastQuestion (NaiveBayesCode naive) {
        String outputWeka;
        Scanner scan = new Scanner(System.in);
        
        
        System.out.print("Do you want to save the model ? (y/n) : ");
        outputWeka = scan.nextLine();
        while(!outputWeka.equals("y") &&
              !outputWeka.equals("Y") &&
              !outputWeka.equals("n") &&
              !outputWeka.equals("N")) {
            System.out.println("press y/n : ");
            outputWeka = scan.next();
            System.out.println("pil : "+outputWeka);
        }
        if (outputWeka.equals("y") || outputWeka.equals("Y")) {
            System.out.print("Input your filename here: ");
            String filename = scan.nextLine();
            try {
                saveModel(naive, filename);
                System.out.println( "                        _\n" +
                                    "           /(|\n" +
                                    "          (  :\n" +
                                    "         __\\  \\  _____\n" +
                                    "       (____)  `|\n" +
                                    "      (____)|   |\n" +
                                    "       (____).__|\n" +
                                    "        (___)__.|_____");
                System.out.println( "=====================================");
                System.out.println( "=============Model Saved=============");
                System.out.println( "=====================================");
            } catch (Exception e) {
                System.out.println("Problem when save the model : " + e);
            }
        } else {
            System.out.println( "==============================================");
            System.out.println( "=============Not saving the model=============");
            System.out.println( "==============================================");
        }
    }
    
    public static void printCoba(Instances instance) {
        System.out.println("");
        System.out.println("1. first instance : " + instance.firstInstance());
        System.out.println("2. banyaknya atribut :" + instance.numAttributes());
        System.out.println("3. " + instance.attribute(0).numValues());
        System.out.println("4. " + instance.attribute(0).weight());
        System.out.println("5. " + instance.attribute(instance.numAttributes()-1).numValues());
        System.out.println("6. " + instance.get(0));
        System.out.println("7. " + instance.get(0).stringValue(4));
        System.out.println("8. " + instance.numInstances());
        System.out.println("9. " + instance.attribute(instance.numAttributes()-1).numValues());
        System.out.println("10. " + instance.get(1).stringValue(instance.numAttributes()-1));
        System.out.println("11. " + instance.attribute(instance.numAttributes()-1).value(0));
        System.out.println("12. " + instance.attribute(instance.numAttributes()-1).name());
        System.out.println("13. " + instance.numClasses());
        System.out.println("14. Banyaknya kelas yang diuji : " + instance.classIndex());
//        System.out.println("15. " + (String.valueOf(instance.attribute(0).value(34)).equals(String.valueOf(4.3))));
//        System.out.println("16. " + instance);
    }
}
