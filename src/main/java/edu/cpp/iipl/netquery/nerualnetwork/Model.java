package edu.cpp.iipl.netquery.nerualnetwork;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by xing on 5/1/16.
 */
public class Model {

    private static final Logger LOG = LoggerFactory.getLogger(Model.class);

    protected final DataSet TRAIN;
    protected final DataSet TEST;

    public Model(boolean enforceNumStab, final DataSet train, final DataSet test) {

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

        return 0;
    }
}
