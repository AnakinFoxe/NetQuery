package edu.cpp.iipl.netquery.nerualnetwork;

import edu.cpp.iipl.netquery.util.DataLoader;
import edu.cpp.iipl.util.Metric;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Created by xing on 4/23/16.
 */
public class Classification extends Model {

    private static final Logger LOG = LoggerFactory.getLogger(Classification.class);

    public Classification(boolean enforceNumStab, DataSet train, DataSet test) {
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

        LOG.warn("Mean Squared Error: {}", mse);

        double kappa = evalResult(labels, predicts);

        LOG.warn("Kappa: {}", kappa);

        return kappa;
    }

    class Result {
        public double label;
        public double predict;

        public Result(double label, double predict) {
            this.label = label;
            this.predict = predict;
        }
    }

    class ResultSort implements Comparator<Result> {

        @Override
        public int compare(Result o1, Result o2) {
            if (o1.predict < o2.predict)
                return -1;
            else if (o1.predict > o2.predict)
                return 1;
            else
                return 0;
        }
    }

    private double evalResult(double[] labels, double[] predicts) {
        PriorityQueue<Result> results = new PriorityQueue<>(new ResultSort());

        for (int i = 0; i < labels.length; ++i)
            results.offer(new Result(labels[i], predicts[i]));

        int size = results.size();

        int cnt1 = (int) (size * 0.06);
        int cnt2 = (int) (size * 0.198);
        int cnt3 = (int) (size * 0.462);

        List<Integer> intLabels = new ArrayList<>();
        List<Integer> intPredicts = new ArrayList<>();

        while (cnt1-- > 0) {
            Result ret = results.poll();

            intLabels.add(DataLoader.roundUp(ret.label));
            intPredicts.add(1);
        }

        while (cnt2-- > 0) {
            Result ret = results.poll();

            intLabels.add(DataLoader.roundUp(ret.label));
            intPredicts.add(2);
        }

        while (cnt3-- > 0) {
            Result ret = results.poll();

            intLabels.add(DataLoader.roundUp(ret.label));
            intPredicts.add(3);
        }

        while (!results.isEmpty()) {
            Result ret = results.poll();

            intLabels.add(DataLoader.roundUp(ret.label));
            intPredicts.add(4);
        }

        return Metric.quadraticWeightedKappa(intLabels, intPredicts);
    }





}
