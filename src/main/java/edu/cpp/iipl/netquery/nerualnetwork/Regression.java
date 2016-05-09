package edu.cpp.iipl.netquery.nerualnetwork;

import edu.cpp.iipl.util.Metric;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by xing on 4/23/16.
 */
public class Regression extends Model {

    private static final Logger LOG = LoggerFactory.getLogger(Regression.class);

    public Regression(boolean enforceNumStab, DataSet train, DataSet test) {
        super(enforceNumStab, train, test);
    }


    @Override
    public double testModel(MultiLayerNetwork net) {
        if (net == null) {
            LOG.error("Not valid network");
            return -1;
        }

        LOG.info("generate test result...");

        // get predicted results
        INDArray output = net.output(TEST.getFeatureMatrix(), Layer.TrainingMode.TEST);
        double[] predicts = output.data().asDouble();

        // get labeled results
        double[] labels = new double[predicts.length];
        for (int i = 0; i < labels.length; ++i)
            labels[i] = TEST.getLabels().getDouble(i);

        double mse = Metric.meanSquaredError(labels, predicts);

        LOG.info("Mean Squared Error: {}", mse);

        return mse;
    }


}
