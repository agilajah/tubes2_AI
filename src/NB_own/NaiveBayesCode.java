/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes2ai;

import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Ghifari
 */
public class NaiveBayesCode {
    private AttributeNominal[] atribNom;
    private int countConclusion[];

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
     * Digunakan untuk mencetak frekuensi tiap nilai atribut
     * @param instance instance yang digunakan
     * @param IndexAtrbt indeks dari atribut yang akan dicetak
     * @param indexConclusion indeks dari label kesimpulan yang akan dicetak
     */
    public void printFrequencyEachValueOfAtribut(Instances instance, int IndexAtrbt, int indexConclusion) {
        
        System.out.println("Print for conclusion : " + instance.attribute(instance.numAttributes()-1).value(indexConclusion));
        
        for (int j=0;j<instance.attribute(IndexAtrbt).numValues();j++){
            System.out.println(j + ". "
                    + instance.attribute(IndexAtrbt).value(j)+" = "
                    + atribNom[IndexAtrbt].getCountAtribut(j,indexConclusion));
        }
    }
    
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
    
    public void printModelProbability(Instances instance, int nConclusion) {
        System.out.println("Print Probability Model for " + instance.attribute(instance.numAttributes()-1).value(nConclusion));
        // iterasi tiap atribut
        for (int i=0;i< instance.numAttributes()-1;i++){ 
            System.out.println(i + ". " + instance.attribute(i));
            // iterasi tiap label dalam atribut
            for (int j=0;j<instance.attribute(i).numValues();j++){
                // iterasi tiap kesimpulan
                System.out.println(j + ". " + instance.attribute(i).value(j)
                        + " " + atribNom[i].getAttribObjectType(j, nConclusion));
            }
        }
    }
    
    /**
     * Menjalankan program utama untuk Naive Bayes
     * @param instance instance yang digunakan
     */
    public void run(Instances instance){
        // Inisiasi awal banyaknya atribut dan data yang di assign
        int nConclusion = instance.attribute(instance.numAttributes()-1).numValues() ;
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
        
//        System.out.println("");
//        printModelProbability(instance, 0);
//        printFrequencyEachValueOfAtribut(instance, 0, 0);
    }
    
    
}
