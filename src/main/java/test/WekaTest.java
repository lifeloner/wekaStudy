package test;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;

/**
 * Created by yang on 2017/3/6.
 */
public class WekaTest {
    public static void main(String[] args) {
//        LibSVM classifier = new LibSVM();
//        J48 classifier = new J48();
        NaiveBayes classifier = new NaiveBayes();
        File testFile = new File("F:\\arff\\Code.arff");
        ArffLoader loader = new ArffLoader();
        try {
            loader.setFile(testFile);
            Instances instances = loader.getDataSet();
            instances.setClassIndex(instances.numAttributes() - 1); // 分类属性行数
            int split = (int) (instances.numInstances() * 0.9);
            Instances train = new Instances(instances, 0, split);
            Instances test = new Instances(instances, split, instances.numInstances() - split);
            classifier.buildClassifier(train);
//            Evaluation evaluation = new Evaluation(instances);
//              evaluation.crossValidateModel(mClassifier, testInstances, 10, new Random(1));
//            evaluation.evaluateModel(classifier,test);
//            System.out.println(evaluation.toClassDetailsString());
//            System.out.println(evaluation.toSummaryString());
//            System.out.println(evaluation.toMatrixString());
            int[] num = new int[]{1, 5, 10, 20};
            int[] count = new int[]{0, 0, 0, 0};
            for (int i = 0; i < test.numInstances(); i++) {
                double[][] result = classifier.classifyInstances(test.instance(i));
                for (int j = 0; j < num.length; j++) {
                    for (int k = 0; k < result.length && k < num[j]; k++) {
                        if (result[k][1] == test.instance(i).classValue()) {
                            count[j]++;
                            break;
                        }
                    }
                }
            }
            for (int i = 0; i < num.length; i++) {
                System.out.println(num[i] + "\t" + 1.0 * count[i] /test.numInstances());
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
