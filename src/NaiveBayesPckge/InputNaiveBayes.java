/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package NaiveBayesPckge;

import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author Ghifari
 */
public class InputNaiveBayes {
    
    /**
     * This class is used to read file .arff
     * @param pathFile path of the file
     * @return 
     */
    public Instances readFileUseWeka (String pathFile) {
        
        Instances instance = null;
        
        try {
            System.out.println("read file . . .");
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(pathFile);
            instance = source.getDataSet();
            if(instance.classIndex() == -1) {
                instance.setClassIndex(instance.numAttributes() - 1);
            }
            System.out.println("file " + pathFile + " has been loaded");
        }catch (Exception e) {
            System.out.println("There is a problem when reading .arff file : " + e);
        }
        
        return instance;
    }
}
    