package edu.cpp.iipl.netquery.util;

import edu.cpp.iipl.netquery.Setting;
import edu.cpp.iipl.netquery.model.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by xing on 4/22/16.
 */
public class DataLoader {

    private final Logger LOG = LoggerFactory.getLogger(DataLoader.class);


    private List<Data> allData;

    public DataLoader() {

    }

    public List<Data> getAllData() {
        return allData;
    }

    public void loadData() throws IOException {
        allData = new ArrayList<>();
        String line;

        // load query
        BufferedReader br = new BufferedReader(new FileReader(Setting.DATASET_QUERY));
        while ((line = br.readLine()) != null) {
            Data data = new Data();
            data.setQuery(line.trim());

            allData.add(data);

        }

        if (allData.size() != Setting.DATASET_SIZE) {
            LOG.error("Query data size: {} does not match config {}",
                    allData.size(), Setting.DATASET_SIZE);
            allData.clear();
            return;
        }

        // load title
        int idx = 0;
        br = new BufferedReader(new FileReader(Setting.DATASET_TITLE));
        while ((line = br.readLine()) != null) {
            if (idx >= Setting.DATASET_SIZE) {
                LOG.error("Title data size: {} does not match config {}",
                        allData.size(), Setting.DATASET_SIZE);
                allData.clear();
                return;
            }

            allData.get(idx++).setTitle(line.trim());
        }

        // load description
        idx = 0;
        br = new BufferedReader(new FileReader(Setting.DATASET_DESCRIPTION));
        while ((line = br.readLine()) != null) {
            if (idx >= Setting.DATASET_SIZE) {
                LOG.error("Description data size: {} does not match config {}",
                        allData.size(), Setting.DATASET_SIZE);
                allData.clear();
                return;
            }

            allData.get(idx++).setDescription(line.trim());
        }

        int[] cnt = new int[4];

        // load relevance
        idx = 0;
        br = new BufferedReader(new FileReader(Setting.DATASET_RELEVANCE));
        while ((line = br.readLine()) != null) {
            if (idx >= Setting.DATASET_SIZE) {
                LOG.error("Relevance data size: {} does not match config {}",
                        allData.size(), Setting.DATASET_SIZE);
                allData.clear();
                return;
            }

            String[] items = line.split(",");
            double relevance = Double.parseDouble(items[0]);
            double variance = Double.parseDouble(items[1]);

            allData.get(idx).setRelevance(relevance);
            allData.get(idx).setVariance(variance);
            ++idx;

            ++cnt[roundUp(relevance) - 1];
        }

        System.out.println(cnt[0] + ", " + cnt[1] + ", " + cnt[2] + ", " + cnt[3]);

        LOG.info("Data set loading accomplished.");

    }


    public static Instances convert2Arff(DataSet data, String nameOfDataSet, String path) {
        int numOfFeature = data.numInputs();
        int numOfSample = data.numExamples();

        // define feature and class prototype
        ArrayList<Attribute> prototype = new ArrayList<>();
        for (int i = 0; i < numOfFeature; ++i)
            prototype.add(new Attribute("attr" + i));   // features
        prototype.add(new Attribute("label"));          // class (label)

        // create empty data set
        Instances dataSet = new Instances(nameOfDataSet, prototype, numOfSample);

        // set last one as class (label)
        dataSet.setClassIndex(numOfFeature);

        // convert data from DataSet to ARFF format
        INDArray featureMatrix = data.getFeatureMatrix();
        INDArray labels = data.getLabels();
        for (int i = 0; i < numOfSample; ++i) {
            Instance sample = new SparseInstance(numOfFeature + 1); // include label

            // set feature vale
            for (int j = 0; j < numOfFeature; ++j) {
                double feat = featureMatrix.getDouble(i, j);
                sample.setValue(prototype.get(j), feat);
            }

            // set label value
            sample.setValue(prototype.get(numOfFeature), DataLoader.roundUp(labels.getDouble(i)));

            // add to data set
            dataSet.add(sample);
        }


        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(path));
            bw.write(dataSet.toString());
            bw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return dataSet;
    }


    public static int roundUp(double num) {
        if (num < 1.5)
            return 1;
        else if (num >= 1.5 && num < 2.5)
            return 2;
        else if (num >= 2.5 && num < 3.5)
            return 3;
        else
            return 4;
    }
}
