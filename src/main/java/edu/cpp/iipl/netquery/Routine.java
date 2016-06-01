package edu.cpp.iipl.netquery;

import edu.berkeley.compbio.jlibsvm.SolutionModel;
import edu.cpp.iipl.netquery.model.Data;
import edu.cpp.iipl.netquery.model.ProcessedData;
import edu.cpp.iipl.netquery.nerualnetwork.Classification;
import edu.cpp.iipl.netquery.nerualnetwork.NetworkConfig;
import edu.cpp.iipl.netquery.nerualnetwork.ParameterSearch;
import edu.cpp.iipl.netquery.nerualnetwork.Regression;
import edu.cpp.iipl.netquery.svm.Model;
import edu.cpp.iipl.netquery.svm.SVMConfig;
import edu.cpp.iipl.netquery.util.DataLoader;
import edu.cpp.iipl.netquery.util.FeatureExtractor;
import edu.cpp.iipl.netquery.util.Preprocessor;
import edu.cpp.iipl.tool.feature.Scaling;
import edu.cpp.iipl.util.MapUtil;
import org.canova.api.io.converters.SelfWritableConverter;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;

/**
 * Created by xing on 4/21/16.
 */
public class Routine {

    private static final Logger LOG = LoggerFactory.getLogger(Routine.class);

    private static DataSet train = new DataSet();
    private static DataSet test = new DataSet();

    private static List<Data> loadData() throws IOException {
        DataLoader loader = new DataLoader();
        loader.loadData();

        return loader.getAllData();
    }

    private static List<ProcessedData> preprocess(List<Data> allData) {
        return Preprocessor.preprocess(allData);
    }

    private static List<List<Double>> extractFeatures(List<ProcessedData> allData)
            throws IOException {
        FeatureExtractor extractor = new FeatureExtractor();

        return extractor.extractFeatures(allData);
    }


    private static void write2File(List<List<Double>> features, List<ProcessedData> allData)
            throws IOException {
        BufferedWriter bw = new BufferedWriter(new FileWriter(Setting.DATASET_FEATURE, false));

        if (features.size() != allData.size()) {
            LOG.error("Feature size {} and Data size {} does not match",
                    features.size(), allData.size());
        }

        for (int i = 0; i < features.size(); ++i) {
            StringBuilder sb = new StringBuilder();

            // concatenate features
            for (Double feature : features.get(i))
                sb.append(feature).append(",");

            // add relevance at last
            sb.append(allData.get(i).getRelevance()).append("\n");

            // write to file
            bw.write(sb.toString());
        }

        bw.close();

        LOG.warn("Write features to the file.");

    }


    private static void loadDataSet(int seed, int sampleSize, int featureNum, double splitRatio) {
        int splitTrainAndEvalNum = (int) (sampleSize * splitRatio);
        int outputNum = 1; // regression has only "1" output


        LOG.warn("load data...");
        RecordReader recordReader = new CSVRecordReader(0, ",");
        try {
            recordReader.initialize(new FileSplit(new File(Setting.DATASET_FEATURE)));
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
            LOG.error("load data failed");
            return;
        }

        DataSetIterator iterator = new RecordReaderDataSetIterator(
                recordReader,
                new SelfWritableConverter(),
                sampleSize,
                featureNum,
                outputNum,
                true);
        DataSet data = iterator.next();

        LOG.warn("split data...");
        SplitTestAndTrain split = data.splitTestAndTrain(splitTrainAndEvalNum, new Random(seed));
        train = split.getTrain();
        test = split.getTest();

    }

    private static NetworkConfig buildNetworkConfigFromFile(String path)
            throws IOException {
        // read config file
        BufferedReader br = new BufferedReader(new FileReader(path));
        List<String> settings = new ArrayList<>();
        String line;
        while ((line = br.readLine()) != null) {
            String[] items = line.trim().split("=");
            settings.add(items[1]);
        }
        br.close();

        // build network config
        int idx = 0;
        NetworkConfig nc = new NetworkConfig();
        nc.type = NetworkConfig.NetworkType.valueOf(settings.get(idx++));
        nc.seed = Integer.parseInt(settings.get(idx++));
        nc.learningRate = Double.parseDouble(settings.get(idx++));
        nc.inNum = Setting.FEATURE_NUM;
        nc.outNum = Integer.parseInt(settings.get(idx++));
        nc.useRegularization = Boolean.parseBoolean(settings.get(idx++));
        nc.l1 = Double.parseDouble(settings.get(idx++));
        nc.l2 = Double.parseDouble(settings.get(idx++));
        nc.numOfNodesInLayer1 = Integer.parseInt(settings.get(idx++));
        nc.numOfNodesInLayer2 = Integer.parseInt(settings.get(idx++));
        nc.numOfNodesInLayer3 = Integer.parseInt(settings.get(idx++));
        nc.numOfNodesInLayer4 = Integer.parseInt(settings.get(idx++));
        nc.afInLayer0 = settings.get(idx).equals("null") ? null : settings.get(idx); ++idx;
        nc.afInLayer1 = settings.get(idx).equals("null") ? null : settings.get(idx); ++idx;
        nc.afInLayer2 = settings.get(idx).equals("null") ? null : settings.get(idx); ++idx;
        nc.afInLayer3 = settings.get(idx).equals("null") ? null : settings.get(idx); ++idx;
        nc.afInOutput = settings.get(idx).equals("null") ? null : settings.get(idx);

        return nc;
    }

    public static void prepareTextForWord2Vec(List<ProcessedData> allData)
            throws IOException {
        BufferedWriter bw = new BufferedWriter(new FileWriter(Setting.CORPUS_CROWDFLOWER));

        Map<String, Integer> wordCount = new HashMap<>();

        // convert to word2vec input text
        for (ProcessedData data : allData) {
            StringBuilder sb = new StringBuilder();

            for (String word : data.getUnigramTitle()) {
                sb.append(word).append(" ");

                MapUtil.updateMap(wordCount, word);
            }

            for (String word : data.getUnigramDesc()) {
                sb.append(word).append(" ");

                MapUtil.updateMap(wordCount, word);
            }

            bw.write(sb.toString());
        }

        // sort word count map
        int top = 1;
        Map<String, Integer> sorted = MapUtil.sortByValue(wordCount);
        for (String word : sorted.keySet()) {
            LOG.warn("top {} word in corpus {} = {}", top, word, sorted.get(word));

            if (++top > 10)
                break;
        }

        bw.close();
    }

    private static void prepareDataSet() throws IOException {
        // load data set
        List<Data> allData = loadData();

        // text pre-processing
        List<ProcessedData> procAllData = preprocess(allData);

        prepareTextForWord2Vec(procAllData);

        // feature extraction
        List<List<Double>> features = extractFeatures(procAllData);

        // feature scaling
        List<List<Double>> scaledFeatures = Scaling.scaleFeatures(features);

        // feature selection

        // write to file (will be load in model training)
        write2File(scaledFeatures, procAllData);

        // update settings
        Setting.SAMPLE_SIZE = scaledFeatures.size();
        Setting.FEATURE_NUM = scaledFeatures.get(0).size();

        // prepare for training & testing
        loadDataSet(Setting.RANDOM_SEED,
                Setting.SAMPLE_SIZE,
                Setting.FEATURE_NUM,
                Setting.SPLIT_RATIO);
    }


    public static void main(String[] args) throws IOException {
        if (args == null || (args.length < 3 || args.length > 5)) {
            LOG.error("Format: [algo] [type] [grid] [file] [iter]");
            LOG.error("     [algo]: nn, svm");
            LOG.error("     [type]: type for nn: classification, regression");
            LOG.error("             type for svm: c_svc, nu_svc, one_class, epsilon_svr, nu_svr");
            LOG.error("     [grid]: true: use grid search");
            LOG.error("             false: use single model");
            LOG.error("     following only works for single model");
            LOG.error("     [file]: single model configuration file");
            LOG.error("     [iter]: iternation number for nn");

            return;
        }

        String algo = args[0];
        String type = args[1];
        boolean useGridSearch = Boolean.parseBoolean(args[2]);
        String path = useGridSearch ? "" : args[3];
        int iteration = 0;
        if (args.length == 5) {
            try {
                iteration = Integer.parseInt(args[4]);
            } catch (NumberFormatException e) {
                LOG.error("Invalid input for iteration");
                LOG.error("", e);
                return;
            }
        }

        LOG.warn("ALGO: {}, TYPE: {}, GRID: {}, PATH: {}, ITER: {}",
                algo, type, useGridSearch, path, iteration);

        // prepare training and testing data
        File trainFile = new File(Setting.DATASET_TRAIN);
        File testFile = new File(Setting.DATASET_TEST);

        if (trainFile.exists() && testFile.exists()) {
            LOG.warn("loading already processed training & testing dataset...");
            train.load(trainFile);
            test.load(testFile);
        } else
            prepareDataSet();

        // save data
        train.save(new File(Setting.DATASET_TRAIN));
        test.save(new File(Setting.DATASET_TEST));

        // train and test model according to arguments
        // neural network
        if (algo.equals("nn")) {
            // regression
            if (type.equals("regression")) {
                // parameter search (it's actually not grid search)
                if (useGridSearch)
                    ParameterSearch.RegressionSearch(train, test);
                // single model
                else {
                    NetworkConfig nc = buildNetworkConfigFromFile(path);

                    Regression nn = new Regression(Setting.NUMERICAL_STABILITY, train, test);

                    MultiLayerNetwork model = nn.trainModel(nc, iteration);

                    System.out.println("MSE of the specific model: " + nn.testModel(model));
                }
            // classification
            } else if (type.equals("classification")) {
                // TODO: only single model mode is supported atm
                NetworkConfig nc = buildNetworkConfigFromFile(path);

                Classification nn = new Classification(Setting.NUMERICAL_STABILITY, train, test);

                MultiLayerNetwork net = nn.trainModel(nc, iteration);

                System.out.println("Kappa of the spefici model: " + nn.testModel(net));

            }
        } else if (algo.equals("svm")) {
            Model svm = new Model(train, test);

            // prepare config
            SVMConfig config = new SVMConfig();

            type = type.toLowerCase();
            if (type.equals("epsilon_svr") || type.equals("nu_svr"))
                config.numOfClass = 0;
            else if (type.equals("c_svc") || type.equals("nu_svc") || type.equals("one_class"))
                config.numOfClass = 4;  // TODO: do not hardcode it
            else {
                LOG.error("Invalid input for SVM type {}", type);
                return;
            }

            // set SVM type
            config.svmType = SVMConfig.SVMType.valueOf(type.toUpperCase());

            // train
            SolutionModel model = svm.trainModel(config, useGridSearch);

            // test
            System.out.println("MSE of the specific model: " + svm.testModel(model));
        }


        Model.runSVMModel(train, test, true);

    }
}
