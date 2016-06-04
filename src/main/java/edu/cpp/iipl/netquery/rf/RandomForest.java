package edu.cpp.iipl.netquery.rf;

import edu.cpp.iipl.netquery.Setting;
import edu.cpp.iipl.netquery.util.DataLoader;
import edu.cpp.iipl.util.Metric;
import jsat.ARFFLoader;
import jsat.DataSet;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by xing on 6/2/16.
 */
public class RandomForest {

    private static final Logger LOG = LoggerFactory.getLogger(RandomForest.class);

    private Instances TRAIN;
    private Instances TEST;

    public RandomForest() {

    }

    public RandomForest(String train, String test) throws Exception {

        this.TRAIN = (new DataSource(train)).getDataSet();
        this.TEST = (new DataSource(test)).getDataSet();

        if (TRAIN.classIndex() == -1)
            TRAIN.setClassIndex(TRAIN.numAttributes() - 1);
        if (TEST.classIndex() == -1)
           TEST.setClassIndex(TEST.numAttributes() - 1);
    }

    public void run() {
        weka.classifiers.trees.RandomForest rf =
                new weka.classifiers.trees.RandomForest();

        try {
            // set parameters
            String[] options = Utils.splitOptions(
                    "-P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1"
            );
            rf.setOptions(options);
            rf.setNumFeatures(100);
            rf.setCalcOutOfBag(true);

            // train random forest model
            rf.buildClassifier(TRAIN);

            // test model
            double[] predicts = new double[TEST.size()];
            double[] labels = new double[TEST.size()];
            for (int i = 0; i < TEST.size(); ++i) {
                double predict = rf.classifyInstance(TEST.get(i));

                predicts[i] = DataLoader.roundUp(predict);
                labels[i] = TEST.get(i).classValue();
            }

            double mse = Metric.meanSquaredError(labels, predicts);
            LOG.warn("Mean Squared Error: {}", mse);

            double kappa = evalResult(labels, predicts);
            LOG.warn("Kappa: {}", kappa);

            Evaluation eval = new Evaluation(TRAIN);
            eval.evaluateModel(rf, TEST);

            System.out.println(eval.toSummaryString());

        } catch (Exception e) {
            LOG.error("{}", e);
        }

    }


    public void runJSAT() {
        File fileTrain = new File(Setting.DATASET_TRAIN_ARFF);
        DataSet trainData = ARFFLoader.loadArffFile(fileTrain);

        File fileTest = new File(Setting.DATASET_TEST_ARFF);
        DataSet testData = ARFFLoader.loadArffFile(fileTest);

        ClassificationDataSet train = new ClassificationDataSet(trainData, 0);
        ClassificationDataSet test = new ClassificationDataSet(testData, 0);

        jsat.classifiers.trees.RandomForest rf =
                new jsat.classifiers.trees.RandomForest();
        rf.setFeatureSamples(100);
        rf.setUseOutOfBagError(true);
        rf.setUseOutOfBagImportance(true);

        // train
        rf.trainC(train);

        // test
        double[] predicts = new double[test.getSampleSize()];
        double[] labels = new double[test.getSampleSize()];
        for (int i = 0; i < test.getSampleSize(); ++i) {
            DataPoint dataPoint = test.getDataPoint(i);

            // predict
            CategoricalResults predict = rf.classify(dataPoint);
            predicts[i] = predict.mostLikely();

            // label
            labels[i] = test.getDataPointCategory(i);
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
