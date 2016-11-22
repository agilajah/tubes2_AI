/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package NaiveBayesPckge;

import java.io.Serializable;

/**
 *
 * @author Ghifari
 */
public class AttributeNominal implements Serializable{
    // Untuk menyimpan nilai probabilitas dari kelas yang diuji
    private Object atribut[][];
    // Untuk menyimpan banyaknya data kelas yang diuji
    private int countAtribut[][];
    
    /**
     * Make Object Biasa
     */
    public AttributeNominal() {
    
    }
    
    /**
     * Membuat object dengan parameter
     * @param nDataDifferent yaitu label (isinya atribut)
     * @param nConclusion yaitu banyaknya konklusi. Misal ada T dan F. berati ada 2
     */
    public AttributeNominal(int nDataDifferent, int nConclusion) {
        atribut = new Object[nDataDifferent][nConclusion];
        countAtribut = new int[nDataDifferent][nConclusion];
    }
    
    /**
     * Mengembalikan atribut dalam bentuk array
     * @return atribut
     */
    public Object[] getAttribArrayObjectType(){
        return atribut;
    }
    
    public void setAtribut(int nDataDifferent, int nConclusion, Double data){
        atribut[nDataDifferent][nConclusion] = data;
    }
    
    public Object getAttribObjectType(int nDataDifferent, int nConclusion){
        return atribut[nDataDifferent][nConclusion];
    }
    
    public void setCountAtribut(int nDataDifferent, int nConclusion, int value) {
        countAtribut[nDataDifferent][nConclusion] = value;
    }
    
    public int getCountAtribut(int index, int nConclusion) {
        return countAtribut[index][nConclusion];
    }
}
