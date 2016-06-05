package edu.cpp.iipl.netquery.rf;

import edu.cpp.iipl.util.Metric;
import jsat.ARFFLoader;
import jsat.DataSet;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by xing on 6/2/16.
 */
public class RandomForest {

    private static final Logger LOG = LoggerFactory.getLogger(RandomForest.class);

    private final ClassificationDataSet TRAIN;
    private final ClassificationDataSet TEST;

    public RandomForest(String train, String test) throws Exception {
        DataSet trainSet = ARFFLoader.loadArffFile(new File(train));
        DataSet testSet = ARFFLoader.loadArffFile(new File(test));

        TRAIN = new ClassificationDataSet(trainSet, 0);
        TEST = new ClassificationDataSet(testSet, 0);
    }


    public void run() {
        LOG.warn("create Random Forest classifier...");
        jsat.classifiers.trees.RandomForest rf =
                new jsat.classifiers.trees.RandomForest();

        // set parameters
        LOG.warn("set up parameters...");
        rf.setFeatureSamples(TRAIN.getNumFeatures()/2);
        rf.setUseOutOfBagError(true);
        rf.setUseOutOfBagImportance(true);

        // train
        LOG.warn("train classifier...");
        rf.trainC(TRAIN);

        // test
        LOG.warn("generate test results...");
        double[] predicts = new double[TEST.getSampleSize()];
        double[] labels = new double[TEST.getSampleSize()];
        for (int i = 0; i < TEST.getSampleSize(); ++i) {
            DataPoint dataPoint = TEST.getDataPoint(i);

            // predict
            CategoricalResults predict = rf.classify(dataPoint);
            predicts[i] = predict.mostLikely();

            // label
            labels[i] = TEST.getDataPointCategory(i);
        }

        double mse = Metric.meanSquaredError(labels, predicts);
        LOG.warn("Mean Squared Error: {}", mse);

        double kappa = evalResult(labels, predicts);
        LOG.warn("Kappa: {}", kappa);

    }


    private double evalResult(double[] labels, double[] predicts) {


        List<Integer> intLabels = new ArrayList<>();
        List<Integer> intPredicts = new ArrayList<>();

        for (int i = 0; i < labels.length; ++i) {
            intLabels.add((int)labels[i]);
            intPredicts.add((int)predicts[i]);
        }

        return Metric.quadraticWeightedKappa(intLabels, intPredicts);
    }

}
