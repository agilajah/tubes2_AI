/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package NaiveBayesPckge;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Ghifari
 */
public class NaiveBayesCode extends AbstractClassifier {
    private AttributeNominal[] atribNom;
    private int countConclusion[];
    private float [] probabConclusion;
    private Instances instanceCopy;
    
    /**
     * Constructor with default value 10
     */
    public NaiveBayesCode() {
        atribNom = new AttributeNominal[10];
    }

    /**
     * Constructor set array with the value of n
     * @param n 
     */
    public NaiveBayesCode(int n) {
        atribNom = new AttributeNominal[n];
    }
    
    /**
     * Untuk membuat tabel banyaknya kesimpulan tiap label. Misal T ada berapa, F ada berapa.
     * @param instance instance yang digunakan
     * @param nConclusion banyaknya label kesimpulan
     */
    public void countConclusionProcedure (Instances instance, int nConclusion) {
        int count;
        for (int index=0;index<nConclusion;index++){
            count = 0;
            for (int i=0;i<instance.numInstances();i++){
                if (instance.get(i).stringValue(instance.numAttributes()-1)
                        == instance.attribute(instance.numAttributes()-1).value(index))
                    count++;
            }
            countConclusion[index] = count;
        }
    }
    
    public void printTheNumOFConclusion(Instances instance, int nConclusion){
        for (int i=0;i<nConclusion;i++) {
            System.out.println(i + ". " + countConclusion[i]);
        }
    }
    
    /**
     * Digunakan untuk menghitung frekuensi tiap nilai atribut
     * @param instance instance yang diberikan
     * @param indexAtrib indeks dari atribut (misal atr1, atr2, atr2)
     * @param indexValue indeks dari value / label dari atribut tertentu
     * @param indexConclusion indeks dari label kesimpulan
     * @return 
     */
    public int countLabel(Instances instance, int indexAtrib, int indexValue, int indexConclusion) {
        int count=0;
        for (int i=0;i<instance.numInstances();i++){
            if ((instance.get(i).stringValue(indexAtrib) == instance.attribute(indexAtrib).value(indexValue))//)
                    && (instance.get(i).stringValue(instance.numAttributes()-1)
                            .equals(instance.attribute(instance.numAttributes()-1).value(indexConclusion))))
                count++;
        }
        return count;        
    }
    
    /**
     * Digunakan untuk menghitung frekuensi tiap nilai atribut dan memanggil fungsi countLabel()
     * @param instance Instance yang digunakan
     */
    public void setCountLabel(Instances instance) {
        // iterasi tiap atribut
        for (int i=0;i< instance.numAttributes()-1;i++){ // issue : why dikurangi satu || Solved. Yes thats right !!
            // iterasi tiap label dalam atribut
            for (int j=0;j<instance.attribute(i).numValues();j++){
                // iterasi tiap kesimpulan
                for (int k=0;k<instance.attribute(instance.numAttributes()-1).numValues();k++){
                    atribNom[i].setCountAtribut(j, k, countLabel(instance, i, j, k));
                }
            }
        }
        
    }
    
    /**
     * Digunakan untuk mencetak frekuensi tiap nilai atribut sesuai dengan indexConclusion
     * @param instance instance yang digunakan
     * @param IndexAtrbt indeks dari atribut yang akan dicetak
     * @param indexConclusion indeks dari label kesimpulan yang akan dicetak
     */
    public void printFrequencyEachValueOfAtributByIndex(Instances instance, int IndexAtrbt, int indexConclusion) {
        
        System.out.println("\nPrint for conclusion : " + instance
                .attribute(instance.numAttributes()-1).value(indexConclusion) + 
                "\natribut : " + instance.attribute(IndexAtrbt).name());
        
        for (int j=0;j<instance.attribute(IndexAtrbt).numValues();j++){
            System.out.println(j + ". "
                    + instance.attribute(IndexAtrbt).value(j)+" = "
                    + atribNom[IndexAtrbt].getCountAtribut(j,indexConclusion));
        }
    }
    
//    public void printFrequencyEachValueOfAtribut()
    
    /**
     * Hitung tiap probabilitas dari tiap label - kesimpulan
     * @param instance instance yang digunakan
     * @param atribNom indeks dari atribut (misal atr1, atr2, atr2)
     * @param indexLabel indeks dari value / label dari atribut tertentu
     * @param nConclusion indeks dari label kesimpulan
     * @return 
     */
    public Double countProbability(Instances instance, int atribNom, int indexLabel, int nConclusion) {
        double result = 0;
        result = (double) this.atribNom[atribNom].getCountAtribut(indexLabel, nConclusion)
                / (double) countConclusion[nConclusion];
//        System.out.println("nConclusion = " + instance.attribute(instance.numAttributes()-1).value(nConclusion) );
//        System.out.println((double) this.atribNom[atribNom].getCountAtribut(indexLabel, nConclusion) +
//                "/" + (double) countConclusion[nConclusion] + 
//                " = " + result);
        return result;
    }
    
    /**
     * Prosedur ini digunakan untuk menghitung Model Probabilitasnya
     * @param instance instance yang digunakan
     */
    public void setModelProbability(Instances instance) {
        // iterasi tiap atribut
        for (int i=0;i< instance.numAttributes()-1;i++){ 
            // iterasi tiap label dalam atribut
            for (int j=0;j<instance.attribute(i).numValues();j++){
                // iterasi tiap kesimpulan
                for (int k=0;k<instance.attribute(instance.numAttributes()-1).numValues();k++){
                    atribNom[i].setAtribut(j, k, countProbability(instance, i, j, k));
                }
            }
        }
    }
    
    public void printModelProbability(Instances instance) {
        System.out.println("Print Probability Model");
        
        // iterasi tiap atribut
        for (int i=0;i< instance.numAttributes()-1;i++){ 
            System.out.println((i+1) + ". " + instance.attribute(i).name());
            
            
            // iterasi tiap label dalam atribut
            for (int j=0;j<instance.attribute(i).numValues();j++){
                if ((j+1)<10)
                    System.out.print("00" + (j+1) + ". " + instance.attribute(i).value(j));
                else if ((j+1)>=10 && j < 100)
                    System.out.print("0" + (j+1) + ". " + instance.attribute(i).value(j));
                else 
                    System.out.print((j+1) + ". " + instance.attribute(i).value(j));
                
                // iterasi tiap kesimpulan
                for (int k=0;k<instance.attribute(instance.numAttributes()-1).numValues();k++){
                    System.out.print("\t\t" + atribNom[i].getAttribObjectType(j, k));
                }
                System.out.println("");
            }
            System.out.println("");
        }
    }
    
    public int searchIndexLabel(int indexAtr, String values) {
        int indeks = -1;
        int numInstance = instanceCopy.numAttributes();
        boolean cek = false;
        
        while (indeks<numInstance && !cek) {
            indeks++;
//            System.out.print("indeksAtr : " + indexAtr + " ");
//            System.out.println("indeks "+indeks + " values = " + values + " instancecopy: " + 
//                    instanceCopy.attribute(indexAtr).value(indeks));
            if (String.valueOf(instanceCopy.attribute(indexAtr).value(indeks))
                    .equals(String.valueOf(values)))
                cek = true;
        }
//        System.out.println(indeks + "indeksssss = "+values);
        return indeks;
    }
    
    int jum=1;
    
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        // banyaknya kesimpulan. Misal T dan F berati ada 2
        int numClasses = instance.numClasses();
        double[] out = new double[numClasses];
        //banyaknya kelas yang diuji
        int class_index = instance.classIndex();
        //banyaknya atribut
        int num_attributes = instance.numAttributes();
        double inputs[] = new double[num_attributes];
//        System.out.println(jum + ". dataset : "+instance.toString());
        
        jum++;
        
        for (int i=0;i<numClasses;i++){
            
            out[i]= probabConclusion[i];
            
            for(int j=0; j<num_attributes-1; j++) {
                int indexLabel = searchIndexLabel(j, instance.stringValue(j));
//                System.out.println("indexLabel =  " + indexLabel);
                out[i] *= (double)atribNom[j].getAttribObjectType(indexLabel, i);
//                System.out.println("nilainya = " + atribNom[j].getAttribObjectType(indexLabel, i));
                
            }
//            System.out.println("out["+i+"] = "+out[i]);
        }
        
        return out;
    }

    
    @Override
    public void buildClassifier(Instances instance) throws Exception {
        run(instance);
        
        //banyaknya instance
        int nInstance = instance.numInstances();
        //banyaknya jenis kesimpulan
        int nProbabConclusion = instance.attribute(instance.numAttributes()-1).numValues();        
        //array untuk menyimpan nilai probabilitas dari konklusi.
        //misal T probabilitasnya berapa, F probabilitasnya berapa.
        probabConclusion = new float[nProbabConclusion];
        for(int i=0;i<nProbabConclusion;i++){
            probabConclusion[i] = (float) countConclusion[i] / (float) nInstance;
        }
        
        Evaluation eval = new Evaluation(instance);
        eval.evaluateModel(this, instance);
        
    }
    
    /**
     * Menjalankan program utama untuk Naive Bayes
     * @param instance instance yang digunakan
     */
    public void run(Instances instance){
        // Inisiasi awal banyaknya atribut dan data yang di assign
        // misal ada T dan F,, berati nConclusion=2
        int nConclusion = instance.attribute(instance.numAttributes()-1).numValues(); // banyaknya hasil konklusi
        instanceCopy = instance;
        for (int i=0;i<instance.numAttributes();i++){
            atribNom[i] = new AttributeNominal(
                         instance.attribute(i).numValues(),
                         nConclusion);
        }
        countConclusion = new int[nConclusion];
        countConclusionProcedure(instance, nConclusion);
                
        
        // hitung jumlah data tiap label di atribut
        setCountLabel(instance);
        // melakukan setting atau mengisi model probabilitas
        setModelProbability(instance);
        
        
        printModelProbability(instance);
//        printFrequencyEachValueOfAtributByIndex(instance, 3, 0);
        System.out.println("");
    }
    
}
