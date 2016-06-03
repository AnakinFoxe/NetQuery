package edu.cpp.iipl.netquery.rf;

import edu.cpp.iipl.netquery.Setting;
import edu.cpp.iipl.netquery.util.DataLoader;
import edu.cpp.iipl.util.Metric;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by xing on 6/2/16.
 */
public class RandomForest {

    private static final Logger LOG = LoggerFactory.getLogger(RandomForest.class);

    private Instances TRAIN;
    private Instances TEST;

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
//            rf.setNumFeatures((int)Math.sqrt(TRAIN.numAttributes()));

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
