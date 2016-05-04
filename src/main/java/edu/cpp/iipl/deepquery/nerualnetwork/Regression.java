package edu.cpp.iipl.deepquery.nerualnetwork;

import edu.cpp.iipl.util.Metric;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by xing on 4/23/16.
 */
public class Regression {

    private static final Logger LOG = LoggerFactory.getLogger(Regression.class);

    private final DataSet TRAIN;
    private final DataSet TEST;


    public Regression(boolean enforceNumStab, final DataSet train, final DataSet test) {

        Nd4j.ENFORCE_NUMERICAL_STABILITY = enforceNumStab;

        this.TRAIN = train;
        this.TEST = test;

    }

    public MultiLayerNetwork trainModel(NetworkConfig nc, int iteration) {
        LOG.info("Start model training...");
        LOG.info("build network...");
        // network configuration
        MultiLayerConfiguration config = NetworkBuilder.getNetworkBuilder(nc, iteration);
        if (config == null) {
            LOG.error("Error configuring network");
            return null;
        }
        // build network
        MultiLayerNetwork net = new MultiLayerNetwork(config);
        net.init();
        net.setListeners(new ScoreIterationListener(1));

        LOG.info("train network...");
        net.fit(TRAIN);

        return net;
    }


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

        double mse = Metric.MeanSquaredError(labels, predicts);

        LOG.info("Mean Squared Error: {}", mse);

        return mse;
    }







}
