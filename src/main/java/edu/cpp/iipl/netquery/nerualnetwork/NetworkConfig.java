package edu.cpp.iipl.netquery.nerualnetwork;

import edu.cpp.iipl.netquery.Setting;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by xing on 4/24/16.
 */
public class NetworkConfig {

    public enum NetworkType {
        DENSE_1_HIDDEN_LAYER,
        DENSE_2_HIDDEN_LAYER,
        DENSE_3_HIDDEN_LAYER,
        DENSE_4_HIDDEN_LAYER,
        DBN_1_HIDDEN_LAYER,
        DBN_2_HIDDEN_LAYER,
        DBN_3_HIDDEN_LAYER,
        DBN_4_HIDDEN_LAYER
    }

    // keep the result of this network config
    public double resultMse = 0;
    public int resultIteration = 0;

    /**
     * Configurations need to be set
     */
    // basic, must have
    public NetworkType type;

    public int seed;
    public double learningRate;

    public int inNum;
    public int outNum;

    public boolean useRegularization = false;
    public double l1 = 1e-6;
    public double l2 = 1e-4;

    // number of hidden nodes
    public int numOfNodesInLayer1 = 0;
    public int numOfNodesInLayer2 = 0;
    public int numOfNodesInLayer3 = 0;
    public int numOfNodesInLayer4 = 0;

    // activation function
    public String afInLayer0 = null;    // input layer
    public String afInLayer1 = null;
    public String afInLayer2 = null;
    public String afInLayer3 = null;
    public String afInOutput = null;

    public OptimizationAlgorithm optAlgo = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;

    public Updater updater = Updater.NESTEROVS;


    public static List<NetworkConfig> genNetworkConfig(NetworkType type, int inNum, int outNum) {
        switch (type) {
            case DENSE_1_HIDDEN_LAYER:
                return genDense1HiddenLayerConfig(inNum, outNum);
            case DENSE_2_HIDDEN_LAYER:
                return genDense2HiddenLayerConfig(inNum, outNum);
            case DENSE_3_HIDDEN_LAYER:
                return genDense3HiddenLayerConfig(inNum, outNum);
            case DENSE_4_HIDDEN_LAYER:
                return genDense4HiddenLayerConfig(inNum, outNum);
            default:
                return new ArrayList<>();
        }
    }


    public static List<NetworkConfig> genConfigOnPreviousConfig(NetworkConfig nc) {
        if (nc == null)
            return new ArrayList<>();

        NetworkType previousType = nc.type;

        switch (previousType) {
            case DENSE_1_HIDDEN_LAYER:
                return genDense2HiddenLayerConfig(nc);
            case DENSE_2_HIDDEN_LAYER:
                return genDense3HiddenLayerConfig(nc);
            case DENSE_3_HIDDEN_LAYER:
                return genDense4HiddenLayerConfig(nc);
            default:
                return new ArrayList<>();
        }
    }


    private static List<NetworkConfig> genDense1HiddenLayerConfig(int inNum, int outNum) {
        List<NetworkConfig> configs = new ArrayList<>();

        for (double learningRate : Setting.COMMON_LEARNING_RATES)
            for (boolean reg : Setting.COMMON_REGS)
                for (int nNodes1 : Setting.COMMON_NUM_OF_NODES)
//                    for (String af0 : Setting.COMMON_AFS)
                        for (String afOut : Setting.COMMON_AFS_OUTPUT) {
                            NetworkConfig nc = new NetworkConfig();

                            nc.type = NetworkType.DENSE_1_HIDDEN_LAYER;
                            nc.seed = Setting.RANDOM_SEED;
                            nc.learningRate = learningRate;
                            nc.inNum = inNum;
                            nc.outNum = outNum;
                            nc.useRegularization = reg;
                            nc.numOfNodesInLayer1 = nNodes1;
                            nc.afInLayer0 = "relu"; // based on observation relu works the best for 1st layer
                            nc.afInOutput = afOut;

                            configs.add(nc);
                        }


        return configs;
    }

    private static List<NetworkConfig> genDense2HiddenLayerConfig(int inNum, int outNum) {
        List<NetworkConfig> configs = new ArrayList<>();

        for (double learningRate : Setting.COMMON_LEARNING_RATES)
            for (boolean reg : Setting.COMMON_REGS)
                for (int nNodes1 : Setting.COMMON_NUM_OF_NODES)
                    for (int nNodes2 : Setting.COMMON_NUM_OF_NODES)
                        for (String af0 : Setting.COMMON_AFS)
                            for (String af1 : Setting.COMMON_AFS)
                                for (String afOut : Setting.COMMON_AFS_OUTPUT) {
                                    NetworkConfig nc = new NetworkConfig();

                                    nc.type = NetworkType.DENSE_1_HIDDEN_LAYER;
                                    nc.seed = Setting.RANDOM_SEED;
                                    nc.learningRate = learningRate;
                                    nc.inNum = inNum;
                                    nc.outNum = outNum;
                                    nc.useRegularization = reg;
                                    nc.numOfNodesInLayer1 = nNodes1;
                                    nc.numOfNodesInLayer2 = nNodes2;
                                    nc.afInLayer0 = af0;
                                    nc.afInLayer1 = af1;
                                    nc.afInOutput = afOut;

                                    configs.add(nc);
                            }


        return configs;
    }

    // based on a 1 hidden layer config
    private static List<NetworkConfig> genDense2HiddenLayerConfig(NetworkConfig pnc) {
        List<NetworkConfig> configs = new ArrayList<>();

        for (int nNode2 : Setting.COMMON_NUM_OF_NODES)
            for (String af1: Setting.COMMON_AFS) {
                NetworkConfig nc = new NetworkConfig();

                nc.type = NetworkType.values()[pnc.type.ordinal() + 1]; // next type
                nc.seed = pnc.seed;
                nc.learningRate = pnc.learningRate;
                nc.inNum = pnc.inNum;
                nc.outNum = pnc.outNum;
                nc.useRegularization = pnc.useRegularization;
                nc.numOfNodesInLayer1 = pnc.numOfNodesInLayer1;
                nc.numOfNodesInLayer2 = nNode2;
                nc.afInLayer0 = pnc.afInLayer0;
                nc.afInLayer1 = af1;
                nc.afInOutput = pnc.afInOutput;

                configs.add(nc);
            }


        return configs;
    }

    private static List<NetworkConfig> genDense3HiddenLayerConfig(int inNum, int outNum) {
        List<NetworkConfig> configs = new ArrayList<>();

        for (double learningRate : Setting.COMMON_LEARNING_RATES)
            for (boolean reg : Setting.COMMON_REGS)
                for (int nNodes1 : Setting.COMMON_NUM_OF_NODES)
                    for (int nNodes2 : Setting.COMMON_NUM_OF_NODES)
                        for (int nNodes3: Setting.COMMON_NUM_OF_NODES)
                            for (String af0 : Setting.COMMON_AFS)
                                for (String af1 : Setting.COMMON_AFS)
                                    for (String af2 : Setting.COMMON_AFS)
                                        for (String afOut: Setting.COMMON_AFS_OUTPUT) {
                                            NetworkConfig nc = new NetworkConfig();

                                            nc.type = NetworkType.DENSE_1_HIDDEN_LAYER;
                                            nc.seed = Setting.RANDOM_SEED;
                                            nc.learningRate = learningRate;
                                            nc.inNum = inNum;
                                            nc.outNum = outNum;
                                            nc.useRegularization = reg;
                                            nc.numOfNodesInLayer1 = nNodes1;
                                            nc.numOfNodesInLayer2 = nNodes2;
                                            nc.numOfNodesInLayer3 = nNodes3;
                                            nc.afInLayer0 = af0;
                                            nc.afInLayer1 = af1;
                                            nc.afInLayer2 = af2;
                                            nc.afInOutput = afOut;

                                            configs.add(nc);
                                        }


        return configs;
    }

    // based on a 2 hidden layer config
    private static List<NetworkConfig> genDense3HiddenLayerConfig(NetworkConfig pnc) {
        List<NetworkConfig> configs = new ArrayList<>();

        for (int nNode3 : Setting.COMMON_NUM_OF_NODES)
            for (String af2: Setting.COMMON_AFS) {
                NetworkConfig nc = new NetworkConfig();

                nc.type = NetworkType.values()[pnc.type.ordinal() + 1]; // next type
                nc.seed = pnc.seed;
                nc.learningRate = pnc.learningRate;
                nc.inNum = pnc.inNum;
                nc.outNum = pnc.outNum;
                nc.useRegularization = pnc.useRegularization;
                nc.numOfNodesInLayer1 = pnc.numOfNodesInLayer1;
                nc.numOfNodesInLayer2 = pnc.numOfNodesInLayer2;
                nc.numOfNodesInLayer3 = nNode3;
                nc.afInLayer0 = pnc.afInLayer0;
                nc.afInLayer1 = pnc.afInLayer1;
                nc.afInLayer2 = af2;
                nc.afInOutput = pnc.afInOutput;

                configs.add(nc);
            }


        return configs;
    }

    private static List<NetworkConfig> genDense4HiddenLayerConfig(int inNum, int outNum) {
        List<NetworkConfig> configs = new ArrayList<>();

        for (double learningRate : Setting.COMMON_LEARNING_RATES)
            for (boolean reg : Setting.COMMON_REGS)
                for (int nNodes1 : Setting.COMMON_NUM_OF_NODES)
                    for (int nNodes2 : Setting.COMMON_NUM_OF_NODES)
                        for (int nNodes3: Setting.COMMON_NUM_OF_NODES)
                            for (int nNodes4 : Setting.COMMON_NUM_OF_NODES)
                                for (String af0 : Setting.COMMON_AFS)
                                    for (String af1 : Setting.COMMON_AFS)
                                        for (String af2 : Setting.COMMON_AFS)
                                            for (String af3: Setting.COMMON_AFS)
                                                for (String afOut : Setting.COMMON_AFS_OUTPUT) {
                                                    NetworkConfig nc = new NetworkConfig();

                                                    nc.type = NetworkType.DENSE_1_HIDDEN_LAYER;
                                                    nc.seed = Setting.RANDOM_SEED;
                                                    nc.learningRate = learningRate;
                                                    nc.inNum = inNum;
                                                    nc.outNum = outNum;
                                                    nc.useRegularization = reg;
                                                    nc.numOfNodesInLayer1 = nNodes1;
                                                    nc.numOfNodesInLayer2 = nNodes2;
                                                    nc.numOfNodesInLayer3 = nNodes3;
                                                    nc.numOfNodesInLayer4 = nNodes4;
                                                    nc.afInLayer0 = af0;
                                                    nc.afInLayer1 = af1;
                                                    nc.afInLayer2 = af2;
                                                    nc.afInLayer3 = af3;
                                                    nc.afInOutput = afOut;

                                                    configs.add(nc);
                                                }


        return configs;
    }

    // based on a 3 hidden layer config
    private static List<NetworkConfig> genDense4HiddenLayerConfig(NetworkConfig pnc) {
        List<NetworkConfig> configs = new ArrayList<>();

        for (int nNode4 : Setting.COMMON_NUM_OF_NODES)
            for (String af3: Setting.COMMON_AFS) {
                NetworkConfig nc = new NetworkConfig();

                nc.type = NetworkType.values()[pnc.type.ordinal() + 1]; // next type
                nc.seed = pnc.seed;
                nc.learningRate = pnc.learningRate;
                nc.inNum = pnc.inNum;
                nc.outNum = pnc.outNum;
                nc.useRegularization = pnc.useRegularization;
                nc.numOfNodesInLayer1 = pnc.numOfNodesInLayer1;
                nc.numOfNodesInLayer2 = pnc.numOfNodesInLayer2;
                nc.numOfNodesInLayer3 = pnc.numOfNodesInLayer3;
                nc.numOfNodesInLayer4 = nNode4;
                nc.afInLayer0 = pnc.afInLayer0;
                nc.afInLayer1 = pnc.afInLayer1;
                nc.afInLayer2 = pnc.afInLayer2;
                nc.afInLayer3 = af3;
                nc.afInOutput = pnc.afInOutput;

                configs.add(nc);
            }


        return configs;
    }
}
