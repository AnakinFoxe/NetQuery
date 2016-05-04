package edu.cpp.iipl.netquery.nerualnetwork;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by xing on 4/24/16.
 */
public class NetworkBuilder {

    private static final Logger LOG = LoggerFactory.getLogger(NetworkBuilder.class);


    public static MultiLayerConfiguration getNetworkBuilder(NetworkConfig nc, int iteration) {
        switch (nc.type) {
            case DENSE_1_HIDDEN_LAYER:
                return getDense1HiddenLayerNet(nc, iteration);
            case DENSE_2_HIDDEN_LAYER:
                return getDense2HiddenLayerNet(nc, iteration);
            case DENSE_3_HIDDEN_LAYER:
                return getDense3HiddenLayerNet(nc, iteration);
            case DENSE_4_HIDDEN_LAYER:
                return getDense4HiddenLayerNet(nc, iteration);
            default:
                LOG.error("Invalid network type {}", nc.type);
                break;
        }

        return null;
    }


    private static MultiLayerConfiguration getDense1HiddenLayerNet(NetworkConfig nc, int iteration) {
        if (nc.numOfNodesInLayer1 == 0
                || nc.afInLayer0 == null || nc.afInOutput == null) {
            LOG.error("Parameter setting invalid for getDense1HiddenLayerNet");
            return null;
        }

        return new NeuralNetConfiguration.Builder()
                .seed(nc.seed)
                .iterations(iteration)
                .optimizationAlgo(nc.optAlgo)
                .learningRate(nc.learningRate)
                .updater(nc.updater)
                .momentum(0.9)
                .regularization(nc.useRegularization)
                .l1(nc.l1)
                .l2(nc.l2)
                .list(2)
                .layer(0, new DenseLayer.Builder()
                        .nIn(nc.inNum)
                        .nOut(nc.numOfNodesInLayer1)
                        .weightInit(WeightInit.XAVIER)
                        .activation(nc.afInLayer0)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(nc.numOfNodesInLayer1)
                        .nOut(nc.outNum)
                        .weightInit(WeightInit.XAVIER)
                        .activation(nc.afInOutput)
                        .build())
                .pretrain(false).backprop(true).build();
    }

    private static MultiLayerConfiguration getDense2HiddenLayerNet(NetworkConfig nc, int iteration) {
        if (nc.numOfNodesInLayer1 == 0 || nc.numOfNodesInLayer2 == 0
                || nc.afInLayer0 == null || nc.afInLayer1 == null || nc.afInOutput == null) {
            LOG.error("Parameter setting invalid for getDense2HiddenLayerNet");
            return null;
        }

        return new NeuralNetConfiguration.Builder()
                .seed(nc.seed)
                .iterations(iteration)
                .optimizationAlgo(nc.optAlgo)
                .learningRate(nc.learningRate)
                .updater(nc.updater)
                .momentum(0.9)
                .regularization(nc.useRegularization)
                .l1(nc.l1)
                .l2(nc.l2)
                .list(3)
                .layer(0, new DenseLayer.Builder()
                        .nIn(nc.inNum)
                        .nOut(nc.numOfNodesInLayer1)
                        .weightInit(WeightInit.XAVIER)
                        .activation(nc.afInLayer0)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(nc.numOfNodesInLayer1)
                        .nOut(nc.numOfNodesInLayer2)
                        .weightInit(WeightInit.XAVIER)
                        .activation(nc.afInLayer1)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(nc.numOfNodesInLayer2)
                        .nOut(nc.outNum)
                        .weightInit(WeightInit.XAVIER)
                        .activation(nc.afInOutput)
                        .build())
                .pretrain(false).backprop(true).build();
    }

    private static MultiLayerConfiguration getDense3HiddenLayerNet(NetworkConfig nc, int iteration) {
        if (nc.numOfNodesInLayer1 == 0 || nc.numOfNodesInLayer2 == 0 || nc.numOfNodesInLayer3 == 0
                || nc.afInLayer0 == null || nc.afInLayer1 == null || nc.afInLayer2 == null
                || nc.afInOutput == null) {
            LOG.error("Parameter setting invalid for getDense3HiddenLayerNet");
            return null;
        }

        return new NeuralNetConfiguration.Builder()
                .seed(nc.seed)
                .iterations(iteration)
                .optimizationAlgo(nc.optAlgo)
                .learningRate(nc.learningRate)
                .updater(nc.updater)
                .momentum(0.9)
                .regularization(nc.useRegularization)
                .l1(nc.l1)
                .l2(nc.l2)
                .list(4)
                .layer(0, new DenseLayer.Builder()
                        .nIn(nc.inNum)
                        .nOut(nc.numOfNodesInLayer1)
                        .weightInit(WeightInit.XAVIER)
                        .activation(nc.afInLayer0)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(nc.numOfNodesInLayer1)
                        .nOut(nc.numOfNodesInLayer2)
                        .weightInit(WeightInit.XAVIER)
                        .activation(nc.afInLayer1)
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .nIn(nc.numOfNodesInLayer2)
                        .nOut(nc.numOfNodesInLayer3)
                        .weightInit(WeightInit.XAVIER)
                        .activation(nc.afInLayer2)
                        .build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(nc.numOfNodesInLayer3)
                        .nOut(nc.outNum)
                        .weightInit(WeightInit.XAVIER)
                        .activation(nc.afInOutput)
                        .build())
                .pretrain(false).backprop(true).build();
    }


    private static MultiLayerConfiguration getDense4HiddenLayerNet(NetworkConfig nc, int iteration) {
        if (nc.numOfNodesInLayer1 == 0 || nc.numOfNodesInLayer2 == 0 || nc.numOfNodesInLayer3 == 0
                || nc.numOfNodesInLayer4 == 0
                || nc.afInLayer0 == null || nc.afInLayer1 == null || nc.afInLayer2 == null
                || nc.afInLayer3 == null || nc.afInOutput == null) {
            LOG.error("Parameter setting invalid for getDense4HiddenLayerNet");
            return null;
        }

        return new NeuralNetConfiguration.Builder()
                .seed(nc.seed)
                .iterations(iteration)
                .optimizationAlgo(nc.optAlgo)
                .learningRate(nc.learningRate)
                .updater(nc.updater)
                .momentum(0.9)
                .regularization(nc.useRegularization)
                .l1(nc.l1)
                .l2(nc.l2)
                .list(5)
                .layer(0, new DenseLayer.Builder()
                        .nIn(nc.inNum)
                        .nOut(nc.numOfNodesInLayer1)
                        .weightInit(WeightInit.XAVIER)
                        .activation(nc.afInLayer0)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(nc.numOfNodesInLayer1)
                        .nOut(nc.numOfNodesInLayer2)
                        .weightInit(WeightInit.XAVIER)
                        .activation(nc.afInLayer1)
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .nIn(nc.numOfNodesInLayer2)
                        .nOut(nc.numOfNodesInLayer3)
                        .weightInit(WeightInit.XAVIER)
                        .activation(nc.afInLayer2)
                        .build())
                .layer(3, new DenseLayer.Builder()
                        .nIn(nc.numOfNodesInLayer3)
                        .nOut(nc.numOfNodesInLayer4)
                        .weightInit(WeightInit.XAVIER)
                        .activation(nc.afInLayer3)
                        .build())
                .layer(4, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(nc.numOfNodesInLayer4)
                        .nOut(nc.outNum)
                        .weightInit(WeightInit.XAVIER)
                        .activation(nc.afInOutput)
                        .build())
                .pretrain(false).backprop(true).build();
    }
}
