package edu.cpp.iipl.netquery;

/**
 * Created by xing on 4/22/16.
 */
public class Setting {

    // data set settings
    public static final String DATASET_TITLE            = "crowdflower/title.csv";
    public static final String DATASET_DESCRIPTION      = "crowdflower/description.csv";
    public static final String DATASET_QUERY            = "crowdflower/query.csv";
    public static final String DATASET_RELEVANCE        = "crowdflower/relevance.csv";
    public static final int DATASET_SIZE                = 20571;

    public static final String DATASET_TRAIN            = "crowdflower/dataset_train";
    public static final String DATASET_TEST             = "crowdflower/dataset_test";

    public static final String CORPUS_CROWDFLOWER       = "crowdflower/corpus.txt";
    public static final String WORD2VEC_VECTORS         = "crowdflower/vectors.txt";


    // pre-processing settings
    public static String TF_REMOVE                      = "html_tag, html_code, url, extra_space";
    public static String TF_FORMAT                      = "email, punctuation, unit, digit";


    // feature extraction settings
    public static boolean INCLUDE_FEAT_COUNT            = true;
    public static boolean INCLUDE_FEAT_OVERLAP          = true;
    public static boolean INCLUDE_FEAT_TFIDF            = false;
    public static boolean INCLUDE_FEAT_WORD2VEC         = true;
    public static boolean INCLUDE_FEAT_WITH_BIGRAM      = true; // not very beneficial
    public static boolean INCLUDE_FEAT_WITH_TRIGRAM     = true; // not very beneficial


    // feature file of the data set
    public static final String DATASET_FEATURE          = "crowdflower/feature.csv";

    public static int SAMPLE_SIZE                       = 0;
    public static int FEATURE_NUM                       = 0;


    // neural network settings
    public static int RANDOM_SEED                       = 55;
    public static double SPLIT_RATIO                    = 0.8;  // split between train and test data
    public static double SELECT_RATIO                   = 0.15; // the ratio to select best network configs
    public static double RESIDUE_RATIO                  = 0.05; // the ratio to randomly add a few more configs
    public static int MAX_SELECT_NUM                    = 20;
    public static int MAX_RANDOM_NUM                    = 5;
    public static boolean NUMERICAL_STABILITY           = true;

    // default parameter settings
    public static final int[] COMMON_SEEDS              = {10, 512, 9527};
    public static final int[] COMMON_ITERATIONS         = {100, 200, 500};
    public static final double[] COMMON_LEARNING_RATES  = {0.5, 0.05, 0.01};
    public static final boolean[] COMMON_REGS           = {false, true};
    public static final int[] COMMON_NUM_OF_NODES       = {50, 100, 250};
    public static final String[] COMMON_AFS             = {"relu", "tanh", "sigmoid", "softmax", "identity"};
    public static final String[] COMMON_AFS_OUTPUT      = {"relu", "identity"};

    public static final int NUM_OF_THREADS              = 8;

    // result
    public static final double MSE_TRESHOLD             = 1;
    public static final double MSE_REDUCED_RATIO        = 0.95;
    public static final String RESULT_FILE              = "crowdflower/result.csv";

}
