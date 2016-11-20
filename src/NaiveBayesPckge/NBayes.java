/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package NaiveBayesPckge;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;
/**
 *
 * @author Ghifari
 */
public class NBayes extends AbstractClassifier {
    private double [][][] counts;
    private Instances instance;
    int nClass = instance.numClasses();
    int nAttrib = instance.numAttributes();
    
    @Override
    public void buildClassifier(Instances instance) throws Exception {
        counts = new double[nClass][nAttrib][0];
        
        
    }
}
