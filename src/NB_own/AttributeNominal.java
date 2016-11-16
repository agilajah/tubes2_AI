/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes2ai;

/**
 *
 * @author Ghifari
 */
public class AttributeNominal {
    private Object atribut[][];
    private int countAtribut[][];
    
    public AttributeNominal() {
    
    }
    
    public AttributeNominal(int nDataDifferent, int nConclusion) {
        atribut = new Object[nDataDifferent][nConclusion];
        countAtribut = new int[nDataDifferent][nConclusion];
    }
    
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
