package edu.cpp.iipl.netquery;

import edu.cpp.iipl.netquery.model.Data;
import edu.cpp.iipl.netquery.model.ProcessedData;
import edu.cpp.iipl.netquery.nerualnetwork.Classification;
import edu.cpp.iipl.netquery.nerualnetwork.NetworkConfig;
import edu.cpp.iipl.netquery.nerualnetwork.ParameterSearch;
import edu.cpp.iipl.netquery.nerualnetwork.Regression;
import edu.cpp.iipl.netquery.util.DataLoader;
import edu.cpp.iipl.tool.feature.Scaling;
import edu.cpp.iipl.tool.feature.extractor.Count;
import edu.cpp.iipl.tool.feature.extractor.Overlap;
import edu.cpp.iipl.tool.preprocessor.SpellCorrector;
import edu.cpp.iipl.tool.preprocessor.Stemmer;
import edu.cpp.iipl.tool.preprocessor.Stopword;
import edu.cpp.iipl.tool.preprocessor.TextFormatter;
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

    private static DataSet train = null;
    private static DataSet test = null;

    private static List<Data> loadData() throws IOException {
        DataLoader loader = new DataLoader();
        loader.loadData();

        return loader.getAllData();
    }

    private static List<ProcessedData> preprocess(List<Data> allData) {
        List<ProcessedData> processedDataList = new ArrayList<>();

        // init text formatter
        Properties props = new Properties();
        props.put("remove", Setting.TF_REMOVE);
        props.put("format", Setting.TF_FORMAT);
        TextFormatter tf = new TextFormatter(props);

        // init spell corrector
        SpellCorrector sc = new SpellCorrector();

        // init stopwords removal
        Stopword sw = new Stopword("en");

        // init stemmer
        Stemmer st = new Stemmer("en");

        for (Data data : allData) {
            // format text
            String query = tf.format(data.getQuery());
            String title = tf.format(data.getTitle());
            String description = tf.format(data.getDescription());

            // convert into words
            List<String> procQuery = Arrays.asList(query.split(" "));
            List<String> procTitle = Arrays.asList(title.split(" "));
            List<String> procDescription = Arrays.asList(description.split(" "));

            // correct misspells
            procQuery = sc.correct(procQuery);
            procTitle = sc.correct(procTitle);
            procDescription = sc.correct(procDescription);

            // remove stopwords
            procQuery = sw.rmStopword(procQuery);
            procTitle = sw.rmStopword(procTitle);
            procDescription = sw.rmStopword(procDescription);

            // stemming
            procQuery = st.stemWords(procQuery);
            procTitle = st.stemWords(procTitle);
            procDescription = st.stemWords(procDescription);

            processedDataList.add(new ProcessedData(
                    procQuery, procTitle, procDescription,
                    data.getRelevance(), data.getVariance()));
        }

        LOG.warn("Preprocessing accomplished.");

        return processedDataList;
    }


    private static List<List<Double>> extractFeatures(List<ProcessedData> allData) {
        List<List<Double>> features = new ArrayList<>();

        if (allData == null || allData.size() == 0)
            return features;

        for (ProcessedData data : allData) {
            List<String> query = data.getQuery();
            List<String> title = data.getTitle();
            List<String> description = data.getDescription();

            List<Double> feature = new ArrayList<>();

            // counting based features
            if (Setting.INCLUDE_FEAT_COUNT) {
                feature.add((double) Count.numOfWords(query));
                feature.add((double) Count.numOfWords(title));
                feature.add((double) Count.numOfWords(description));
            }

            // overlapping based features
            if (Setting.INCLUDE_FEAT_OVERLAP) {
                // complete query in title/description (overlap1)
                int numOfQueryInTitle = Overlap.numOfStr1InStr2(query, title);
                int numOfQueryInDescription = Overlap.numOfStr1InStr2(query, description);

                // last word of query in title/description (overlap2)
                int numOfLastWordQueryInTitle = 0;
                int numOfLastWordQueryInDescription = 0;
                if (query.size() > 0) {
                    String lastWord = query.get(query.size() - 1);
                    numOfLastWordQueryInTitle = Overlap.numOfWordInWords(lastWord, title);
                    numOfLastWordQueryInDescription = Overlap.numOfWordInWords(lastWord, description);
                }

                // all the words of query in title/description (overlap3)
                int numOfWordsQueryInTitle = Overlap.numOfWords1InWords2(query, title);
                int numOfWordsQueryInDescription = Overlap.numOfWords1InWords2(query, description);

                // ratio of (overlap3 / length of title/description)
                double ratioOfTitle = 0;
                double ratioOfDescription = 0;
                if (title.size() > 0)
                    ratioOfTitle = (double) numOfWordsQueryInTitle / title.size();
                if (description.size() > 0)
                    ratioOfDescription = (double) numOfQueryInDescription / description.size();

                feature.add((double) numOfQueryInTitle);
                feature.add((double) numOfQueryInDescription);
                feature.add((double) numOfLastWordQueryInTitle);
                feature.add((double) numOfLastWordQueryInDescription);
                feature.add((double) numOfWordsQueryInTitle);
                feature.add((double) numOfWordsQueryInDescription);
                feature.add(ratioOfTitle);
                feature.add(ratioOfDescription);
            }

            // tf-idf based features
            if (Setting.INCLUDE_FEAT_TFIDF) {

            }

            features.add(feature);
        }

        LOG.warn("Feature extraction accomplished.");

        return features;
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


    private static void runSpecificModel(String[] args, final DataSet train, final DataSet test)
            throws IOException {
        if (args.length != 3) {
            LOG.error("Invalid arguments {}", args);
            return;
        }

        String type = args[0];
        int iteration = Integer.parseInt(args[1]);
        String path = args[2];

        NetworkConfig nc = buildNetworkConfigFromFile(path);

        double ret = 0;

        if (type.equals("classification")) {
            Classification nn = new Classification(Setting.NUMERICAL_STABILITY, train, test);

            MultiLayerNetwork net = nn.trainModel(nc, iteration);

            ret = nn.testModel(net);

            System.out.println("Kappa of the spefici model: " + ret);
        } else if (type.equals("regression")) {
            Regression nn = new Regression(Setting.NUMERICAL_STABILITY, train, test);

            MultiLayerNetwork net = nn.trainModel(nc, iteration);

            ret = nn.testModel(net);

            System.out.println("MSE of the specific model: " + ret);
        } else
            LOG.error("Type {} not supported", type);
    }


    public static void main(String[] args) throws IOException {
        if (args == null || (args.length != 0 && args.length != 3)) {
            LOG.error("Format:");
            LOG.error("  1) no argument: run parameter search");
            LOG.error("  2) [file] [type]");
            LOG.error("     [file]: file of network config");
            LOG.error("     [type]: model type");
            return;
        }

        if (args.length == 3 && (!args[0].equals("classification") && !args[0].equals("regression"))) {
            LOG.error("Model type should be either classification or regression");
            return;
        }

        if (args.length == 3) {
            try {
                int iteration = Integer.parseInt(args[1]);
            } catch (NumberFormatException e) {
                LOG.error("Invalid input for iteration");
                LOG.error("", e);
                return;
            }
        }

        // load data set
        List<Data> allData = loadData();

        // text pre-processing
        List<ProcessedData> procAllData = preprocess(allData);

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

        // default: parameter search
        // other wise: train and test a specific model
        if (args.length == 0)
            // parameter search to find the best model
            ParameterSearch.RegressionSearch(train, test);
        else
            runSpecificModel(args, train, test);


    }
}