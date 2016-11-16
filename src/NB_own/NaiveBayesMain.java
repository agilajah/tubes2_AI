/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes2ai;

import java.util.Enumeration;
import java.util.Scanner;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;


/**
 *
 * @author Ghifari
 */
public class NaiveBayesMain {
    
    //public static void 
    
    public static Instances useFilter(Instances instance){
        boolean isNumeric = false;
        Instances newInstance = null;
        
        for(int i=0; i<instance.numAttributes(); i++) {			
            if(instance.attribute(i).type() == 0) {
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
        inputWeka = "d:/iris.arff";
        instance = nb.readFileUseWeka(inputWeka);
        instance = useFilter(instance);
        naive = new NaiveBayesCode(instance.numAttributes());
        
        naive.run(instance);
        
        
        
        System.out.println("");
        System.out.println("1. " + instance.firstInstance());
        System.out.println("2. " + instance.numAttributes());
        System.out.println("3. " + instance.attribute(4).numValues());
        System.out.println("4. " + instance.attribute(0).weight());
        System.out.println("5. " + instance.attribute(instance.numAttributes()-1).numValues());
        System.out.println("6. " + instance.get(0));
        System.out.println("7. " + instance.get(0).stringValue(4));
        System.out.println("8. " + instance.numInstances());
        System.out.println("9. " + instance.attribute(instance.numAttributes()-1).numValues());
        System.out.println("10. " + instance.get(120).stringValue(instance.numAttributes()-1));
        System.out.println("11. " + instance.attribute(instance.numAttributes()-1).value(2));
        System.out.println("11. " + instance.attribute(0));
        
    }
}
