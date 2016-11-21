/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package NaiveBayesPckge;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Scanner;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToNominal;


/**
 *
 * @author Ghifari
 */
public class NaiveBayesMain {    
    private static Classifier naive;
    private static DecimalFormat df = new DecimalFormat("#.####");
        
    public static Instances useFilterDiscritize(Instances dataSet) throws Exception {
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
    
    public static void saveModel(String filename) throws Exception {
        weka.core.SerializationHelper.write("D:/ITB/" + filename + ".nb", naive);
    }


    public static NaiveBayesCode loadModel(String filename) throws Exception {
        NaiveBayesCode loader = (NaiveBayesCode) weka.core.SerializationHelper.read("D:/ITB/" + filename + ".nb");

        return loader;
    }    
    
    public static void addNewInstance(Instances instances) throws Exception {
        Scanner scan = new Scanner(System.in);
        ArrayList<Attribute> atts = new ArrayList<Attribute>();
        ArrayList<String> classVal = new ArrayList<String>();
        int nConclus = instances.attribute(instances.numAttributes()-1).numValues();
        int numAttribut = instances.numAttributes();
        
        //buat nambah kesimpulan. Misal T dan F
        for (int i=0;i<nConclus;i++){ 
            classVal.add(instances.attribute(instances.numAttributes()-1).value(i));
        }
        
        //buat nambahin attribut
        for (int i=0;i<numAttribut-1;i++){ 
            atts.add(new Attribute(instances.attribute(i).name()));
        }
        atts.add(new Attribute(instances.attribute(numAttribut-1).name(),classVal));

        double[] attValues = new double[numAttribut];
        System.out.print("Masukkan nilai : ");
        for (int i=0; i < numAttribut-1;i++) {
            attValues[i] = scan.nextDouble();
        }
        Discretize discretize = new Discretize();
        String s = scan.nextLine();

        Instance instance = new DenseInstance(1.0, attValues);


        instance.setDataset(instances);
        
        discretize.setInputFormat(instances);
        discretize.input(instance);
        
        int classify1 = (int) naive.classifyInstance(instance);
        System.out.print("Prediction Class : ");
        System.out.println(classVal.get(classify1));
    }
    
    public static void printEvaluation(Instances instance) throws Exception {
        Evaluation eval = new Evaluation(instance);
        eval.evaluateModel(naive, instance);

        System.out.println(eval.toSummaryString());		// Summary of Training
        System.out.println(eval.toMatrixString());

        double errorRates = eval.incorrect() / eval.numInstances() * 100;
        double accuracy = eval.correct() / eval.numInstances() * 100;

        System.out.println("Accuracy: " + df.format(accuracy) + " %");
        System.out.println("Error rate: " + df.format(errorRates) + " %"); // Printing Training Mean root squared error
    }
    
    public static void main(String[] args) {
        Instances instance;
        String filterr ;
        Scanner scan = new Scanner(System.in);
        InputNaiveBayes nb = new InputNaiveBayes();
        String inputWeka;
        
        System.out.println("   Welcome to the Naive Bayes System");
        System.out.println("========================================");
        System.out.println("");
        
        System.out.print("Now input arff path file : ");
        inputWeka = scan.nextLine();
    //    inputWeka = "zdata/iris.arff";
        instance = nb.readFileUseWeka(inputWeka);
        
        try {
            System.out.println("Do you want to use filter ? Please choose one : ");
                System.out.println("  1. Nominal To Numeric");
                System.out.println("  2. Discretize");
                System.out.println("  3. Don't use filter");
                System.out.print("Your answer : ");
                filterr = scan.nextLine();
            while (!filterr.equals("1") &&
                   !filterr.equals("2") &&
                   !filterr.equals("3")){
                System.out.println("Type the number please : ");
                System.out.println("  1. Nominal To Numeric");
                System.out.println("  2. Discretize");
                System.out.println("  3. Don't use filter");
                System.out.print("Your answer : ");
                filterr = scan.nextLine();
            }
            if (filterr.equals("1")) 
                instance = useFilterNominalToNumeric(instance);
            else if (filterr.equals("2")) 
                instance = useFilterDiscritize(instance);
            else {
                System.out.println("> Data is not filtered\n");
            }
        } catch (Exception e) {
            System.out.println("Problem when use filter : " + e);
        }
        
        String choice = firstQuestion(instance);
        if (choice.equals("2")) {
            naive = new NaiveBayesCode(instance.numAttributes());
            try {
                naive.buildClassifier(instance);
            } catch (Exception e) {
                System.out.println("Problem when build classifier : " + e);
            }

            try {
                printEvaluation(instance);
                lastQuestion();
            } catch (Exception e) {
                System.out.println("Problem on evaluation : " + e);
            }
        }
        
        goodByeMessage();
//        try {
//            addNewInstance(instance);
//        } catch (Exception ex) {
//            Logger.getLogger(NaiveBayesMain.class.getName()).log(Level.SEVERE, null, ex);
//        }
    }
    
    public static String firstQuestion(Instances instance) {
        String choice;
        String nameModel;
        Scanner scan = new Scanner(System.in);
        
        System.out.println("Do you want to load the mode or do you want to test the data ?");
        System.out.println("   1. Load model");
        System.out.println("   2. Test the data");
        System.out.print("Which one do you want : ");
        choice = scan.nextLine();
        
        while (!choice.equals("1") && !choice.equals("2")) {
            System.out.println("   1. Load model");
            System.out.println("   2. Test the data");
            System.out.print("Type the number of your decision please : ");
            choice = scan.nextLine();
        }
        
        if (choice.equals("1")) {
            System.out.print("type the name of your model : ");
            nameModel = scan.next();
            try {
                naive = loadModel(nameModel);
                System.out.println( "        ,-\"\"-.\n" +
                                    "       :======:\n" +
                                    "       :======;\n" +
                                    "        `-.,-'\n" +
                                    "          ||\n" +
                                    "        _,''--.    _____\n" +
                                    "       (/ __   `._|\n" +
                                    "      ((_/_)\\     |\n" +
                                    "       (____)`.___|\n" +
                                    "        (___)____.|_____\n" +
                                    "                     SSt");
                System.out.println( "======================================");
                System.out.println( "=============Model Loaded=============");
                System.out.println( "======================================");
            } catch (Exception ex) {
                System.out.println("Problem when load the model : " + ex);
            }
            try {
                printEvaluation(instance);
            } catch (Exception ex) {
                System.out.println("Problem when print the Evaluation at loadModel : " + ex);
            }
        }
        
        return choice;
    }
    
    public static void lastQuestion () {
        String outputWeka;
        Scanner scan = new Scanner(System.in);
        
        System.out.println("");
        System.out.println("");
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
                saveModel(filename);
                System.out.println( "             _\n" +
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
    
    public static void goodByeMessage() {
        System.out.println("");
        System.out.println("");
        System.out.println("");
        System.out.println("================================================");
        System.out.println("======== Thank you for using our system ========");
        System.out.println("");
        System.out.println( "                    |\\ /\\\n" +
                            "    __              |,\\(_\\_\n" +
                            "   ( (              |\\,`   `-^.\n" +
                            "    \\ \\             :    `-'   )\n" +
                            "     \\ \\             \\        ;\n" +
                            "      \\ \\             `-.   ,'\n" +
                            "       \\ \\ ____________,'  (\n" +
                            "        ; '                ;\n" +
                            "        \\                 /___,-.\n" +
                            "         `,    ,_____|  ;'_____,'\n" +
                            "       ,-\" \\  :      | :\n" +
                            "      ( .-\" \\ `.__   | |\n" +
                            "       \\__)  `.__,'  |__)  Byee byee ....");
        System.out.println("================================================");
        System.out.println("================================================");
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
