package edu.cpp.iipl.deepquery.nerualnetwork;

import edu.cpp.iipl.deepquery.Setting;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.*;

/**
 * Created by xing on 4/28/16.
 */
public class ParameterSearch {

    private static final Logger LOG = LoggerFactory.getLogger(ParameterSearch.class);

    static class PQSort implements Comparator<NetworkConfig> {

        @Override
        public int compare(NetworkConfig o1, NetworkConfig o2) {
            if (o1.resultMse < o2.resultMse)
                return -1;
            else if (o1.resultMse > o2.resultMse)
                return 1;
            else
                return 0;
        }
    }


    public static void RegressionSearch(final DataSet train, final DataSet test)
            throws IOException {
        int inNum = Setting.FEATURE_NUM;
        int outNum = 1;

        // init priority queue
        PriorityQueue<NetworkConfig> pq = new PriorityQueue<>(new PQSort());

        // prepare file to write
        BufferedWriter bw = new BufferedWriter(new FileWriter(Setting.RESULT_FILE));


        for (NetworkConfig.NetworkType type : NetworkConfig.NetworkType.values()) {
            // prepare list of network config to process
            List<NetworkConfig> ncList;
            if (type == NetworkConfig.NetworkType.DENSE_1_HIDDEN_LAYER)
                // start over at the base level of a type series
                ncList = NetworkConfig.genNetworkConfig(type, inNum, outNum);
            else
                // generate network configs based on previous good ones
                ncList = genConfigForNextType(pq);

            LOG.warn("Searching {}: {} configs", type, ncList.size());

            // clear priority queue
            pq.clear();

            // create thread pool
            ExecutorService es = Executors.newFixedThreadPool(Setting.NUM_OF_THREADS);

            // process with all the network configs
            List<Future<List<NetworkConfig>>> futureList = new ArrayList<>();
            for (NetworkConfig nc : ncList)
                futureList.add(es.submit(new ModelEvalThread(nc, train, test)));

            // shutdown executors
            es.shutdown();
            try {
                es.awaitTermination(7, TimeUnit.DAYS);
            } catch (InterruptedException e) {
                LOG.error("{}", e);
            }

            // put results into priority queue and write to file
            for (Future<List<NetworkConfig>> future : futureList) {
                try {
                    List<NetworkConfig> results = future.get();

                    double minMse = Double.MAX_VALUE;
                    NetworkConfig bestNc = null;
                    for (NetworkConfig result : results) {
                        if (minMse > result.resultMse) {
                            minMse = result.resultMse;
                            bestNc = result;
                        }

                        // write the result to file
                        bw.write(getResultStr(result));
                    }

                    // save the best network config (no matter with which iteration)
                    if (bestNc != null)
                        pq.offer(bestNc);
                } catch (InterruptedException | ExecutionException e) {
                    LOG.error("{}", e);
                }
            }

            bw.flush();
        }

        bw.close();

    }


    private static String getResultStr(NetworkConfig nc)
            throws IOException {
        StringBuilder sb = new StringBuilder();

        sb.append(nc.resultMse).append(",")
                .append(nc.type).append(",")
                .append(nc.resultIteration).append(",")
                .append(nc.learningRate).append(",")
                .append(nc.inNum).append(",")
                .append(nc.outNum).append(",")
                .append(nc.useRegularization).append(",")
                .append(nc.l1).append(",")
                .append(nc.l2).append(",")
                .append(nc.numOfNodesInLayer1).append(",")
                .append(nc.numOfNodesInLayer2).append(",")
                .append(nc.numOfNodesInLayer3).append(",")
                .append(nc.numOfNodesInLayer4).append(",")
                .append(nc.afInLayer0).append(",")
                .append(nc.afInLayer1).append(",")
                .append(nc.afInLayer2).append(",")
                .append(nc.afInLayer3).append(",")
                .append(nc.afInOutput).append(",")
                .append(nc.optAlgo).append(",")
                .append(nc.updater)
                .append("\n");

        return sb.toString();
    }

    private static List<NetworkConfig.NetworkType> getInitTypes() {
        List<NetworkConfig.NetworkType> types = new ArrayList<>();

        // TODO: add more initial level configs
        types.add(NetworkConfig.NetworkType.DENSE_1_HIDDEN_LAYER);

        return types;
    }

    private static List<List<NetworkConfig>> getAllConfigs(int inNum, int outNum) {
        List<List<NetworkConfig>> ncList = new ArrayList<>();

        for (NetworkConfig.NetworkType type : NetworkConfig.NetworkType.values())
            ncList.add(NetworkConfig.genNetworkConfig(type, inNum, outNum));

        return ncList;
    }

    private static List<List<NetworkConfig>> getInitTypeConfigs(int inNum, int outNum) {
        List<List<NetworkConfig>> ncList = new ArrayList<>();

        for (NetworkConfig.NetworkType type : getInitTypes())
            ncList.add(NetworkConfig.genNetworkConfig(type, inNum, outNum));

        return ncList;
    }

    private static List<NetworkConfig> genConfigForNextType(PriorityQueue<NetworkConfig> pq) {
        List<NetworkConfig> ncList = new ArrayList<>();

        int totalSize = pq.size();

        // retrieve and merge top 20% network configs from priority queue
        // but no more than the maximum number
        int numOfTops = Math.min((int) (totalSize * Setting.SELECT_RATIO + 0.5),
                Setting.MAX_SELECT_NUM);
        for (int i = 0; i < numOfTops; ++i) {
            NetworkConfig pnc = pq.poll();

            ncList.addAll(NetworkConfig.genConfigOnPreviousConfig(pnc));
        }

        Random random = new Random(Setting.RANDOM_SEED);
        // randomly retrieve 10% from rest of the
        // but no more than the maximum number
        int numOfResidue = Math.min((int) (totalSize * Setting.RESIDUE_RATIO + 0.5),
                Setting.MAX_RANDOM_NUM);
        while (numOfResidue > 0 && pq.size() > 0) {
            NetworkConfig pnc = pq.poll();

            if (random.nextInt(10) <= 2) {
                ncList.addAll(NetworkConfig.genConfigOnPreviousConfig(pnc));
                --numOfResidue;
            }
        }

        return ncList;
    }


    static class ModelEvalThread implements Callable<List<NetworkConfig>> {

        private final NetworkConfig nc;
        private final DataSet train;
        private final DataSet test;


        public ModelEvalThread(NetworkConfig nc, DataSet train, DataSet test) {
            this.nc = nc;
            this.train = train;
            this.test = test;
        }

        @Override
        public List<NetworkConfig> call() throws Exception {
            List<NetworkConfig> ncList = new ArrayList<>();

            double minMse = Double.MAX_VALUE;
            for (int iteration : Setting.COMMON_ITERATIONS) {
                Regression nn = new Regression(Setting.NUMERICAL_STABILITY, train, test);

                // model training
                MultiLayerNetwork net = nn.trainModel(nc, iteration);

                // model testing
                double mse = nn.testModel(net);

                // keep result only if it exceeds threshold
                // or the mse reduced certain ratio
                // otherwise, stop computing with more iterations (to save time)
                if (mse >= Setting.MSE_TRESHOLD || (mse / minMse) > Setting.MSE_REDUCED_RATIO)
                    break;
                else {
                    // set the result to network config
                    nc.resultMse = mse;
                    nc.resultIteration = iteration;

                    minMse = mse;

                    // save this network config
                    ncList.add(nc);
                }
            }

            return ncList;
        }
    }


}
