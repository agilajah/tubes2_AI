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
    
    public void setCountLabel(Instances instance) {
        // iterasi tiap atribut
        for (int i=0;i< instance.numAttributes()-1;i++){ // issue : why dikurangi satu
            // iterasi tiap label dalam atribut
            for (int j=0;j<instance.attribute(i).numValues();j++){
                // iterasi tiap kesimpulan
                for (int k=0;k<instance.attribute(instance.numAttributes()-1).numValues();k++){
                    atribNom[i].setCountAtribut(j, k, countLabel(instance, i, j, k));
                }
                
            }
        }
    }
    
    public void printFrequencyEachValueOfAtribut(Instances instance, int IndexAtrbt, int indexConclusion) {
        
        System.out.println("Print for conclusion : " + instance.attribute(instance.numAttributes()-1).value(indexConclusion));
        
        for (int j=0;j<instance.attribute(IndexAtrbt).numValues();j++){
            System.out.println(j + ". "
                    + instance.attribute(IndexAtrbt).value(j)+" = "
                    + atribNom[IndexAtrbt].getCountAtribut(j,indexConclusion));
        }
    }
    
    public void run(Instances instance){
        // Inisiasi awal banyaknya atribut dan data yang di assign
        for (int i=0;i<instance.numAttributes();i++){
            atribNom[i] = new AttributeNominal(
                         instance.attribute(i).numValues(),
                         instance.attribute(instance.numAttributes()-1).numValues());
        }
        
        // hitung jumlah data tiap label di atribut
        setCountLabel(instance);
        
        printFrequencyEachValueOfAtribut(instance, 0, 0);
    }
    
    
}
