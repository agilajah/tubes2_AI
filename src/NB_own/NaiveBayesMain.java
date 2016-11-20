/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package NaiveBayesPckge;

import java.util.Scanner;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToNominal;


/**
 *
 * @author Ghifari
 */
public class NaiveBayesMain {    
    
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
    
    
    
    public static void main(String[] args) {
        Instances instance = null;
        Scanner scan = new Scanner(System.in);
        InputNaiveBayes nb = new InputNaiveBayes();
        NaiveBayesCode naive;
        
        System.out.println("Welcome to the Naive Bayes System");
        
        String inputWeka;
        System.out.print("Input arff path file : ");
        //inputWeka = scan.nextLine();
        inputWeka = "d:/Tennis.arff";
        instance = nb.readFileUseWeka(inputWeka);
        
        try {
//            instance = useFilterNominalToNumeric(instance);
            instance = useFilterDiscritize(instance);
        } catch (Exception e) {
            System.out.println("Problem when use filter : " + e);
        }
        
        naive = new NaiveBayesCode(instance.numAttributes());
        try {
            naive.buildClassifier(instance);
            NaiveBayes nbb = new NaiveBayes();
            Evaluation eval = new Evaluation(instance);
            nbb.buildClassifier(instance);
//            eval.evaluateModel(nbb, instance);
            
            eval.evaluateModel(naive, instance);
            System.out.println(eval.toSummaryString());		// Summary of Training
            System.out.println(eval.toMatrixString());
        } catch (Exception e) {
            System.out.println("Problem : " + e);
        }
//        printCoba(instance);
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
        System.out.println("10. " + instance.get(120).stringValue(instance.numAttributes()-1));
        System.out.println("11. " + instance.attribute(instance.numAttributes()-1).value(2));
        System.out.println("12. " + instance.attribute(0).name());
        System.out.println("13. " + instance.numClasses());
        System.out.println("14. Banyaknya kelas yang diuji" + instance.classIndex());
//        System.out.println("15. " + (String.valueOf(instance.attribute(0).value(34)).equals(String.valueOf(4.3))));
//        System.out.println("16. " + instance);
    }
}
