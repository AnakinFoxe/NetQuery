package edu.cpp.iipl.netquery.svm;

import edu.berkeley.compbio.jlibsvm.*;
import edu.berkeley.compbio.jlibsvm.binary.BinaryClassificationSVM;
import edu.berkeley.compbio.jlibsvm.binary.C_SVC;
import edu.berkeley.compbio.jlibsvm.binary.MutableBinaryClassificationProblemImpl;
import edu.berkeley.compbio.jlibsvm.binary.Nu_SVC;
import edu.berkeley.compbio.jlibsvm.kernel.*;
import edu.berkeley.compbio.jlibsvm.labelinverter.ByteLabelInverter;
import edu.berkeley.compbio.jlibsvm.multi.MultiClassificationSVM;
import edu.berkeley.compbio.jlibsvm.multi.MutableMultiClassProblemImpl;
import edu.berkeley.compbio.jlibsvm.oneclass.OneClassSVC;
import edu.berkeley.compbio.jlibsvm.regression.EpsilonSVR;
import edu.berkeley.compbio.jlibsvm.regression.MutableRegressionProblemImpl;
import edu.berkeley.compbio.jlibsvm.regression.Nu_SVR;
import edu.berkeley.compbio.jlibsvm.scaler.NoopScalingModel;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import edu.cpp.iipl.netquery.util.DataLoader;
import edu.cpp.iipl.util.Metric;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashSet;

/**
 * Created by xing on 5/30/16.
 */
public class Model {

    private static final Logger LOG = LoggerFactory.getLogger(Model.class);


    private final DataSet TRAIN;
    private final DataSet TEST;

    public Model(DataSet train, DataSet test) {
        this.TRAIN = train;
        this.TEST = test;
    }


    public SolutionModel trainModel(SVMConfig config, boolean useGridSearch) {
        // setup parameter
        ImmutableSvmParameter param;

        if (useGridSearch)
            param = setupGridSearchParam(config);
        else
            param = setupSingleModelParam(config);

        // instantiate svm problem
        MutableSvmProblem problem = getProblem(config.numOfClass);

        // instantiate svm object according to type
        SVM svm = getSVM(config.svmType);

        if (svm instanceof BinaryClassificationSVM && config.numOfClass > 2) {
            svm = new MultiClassificationSVM((BinaryClassificationSVM) svm);
        }

        return svm.train(problem, param);
    }


    public double testModel(SolutionModel model) {
        if (model == null) {
            LOG.error("Model is invalid");
            return Double.NaN;
        }

        int testSize = TEST.getFeatureMatrix().rows();
        double[] predicts = new double[testSize];
        double[] labels = new double[testSize];

        for (int i = 0; i < testSize; ++i) {
            // convert testing data
            INDArray row = TEST.getFeatureMatrix().getRow(i);
            int numOfFeatures = row.columns();
            SparseVector vector = new SparseVector(numOfFeatures);
            for (int v = 0; v < numOfFeatures; ++v) {
                vector.indexes[v] = v + 1;
                vector.values[v] = (float)row.getDouble(v);
            }

            // get prediction
            if (model instanceof ContinuousModel)
                predicts[i] = ((ContinuousModel)model).predictValue(vector);
            else
                predicts[i] = (Byte)((DiscreteModel)model).predictLabel(vector);

            // get label
            labels[i] = TEST.getLabels().getDouble(i);
        }

        double mse = Metric.meanSquaredError(labels, predicts);

        LOG.warn("Mean Squared Error: {}", mse);

        return mse;
    }

    public static void runSVMModel(DataSet train, DataSet test, boolean useGridSearch) {
        Model svm = new Model(train, test);

        // prepare config
        SVMConfig config = new SVMConfig();

        // train
        SolutionModel model = svm.trainModel(config, useGridSearch);

        // test
        svm.testModel(model);
    }

    private ImmutableSvmParameterPoint setupSingleModelParam(SVMConfig config) {
        ImmutableSvmParameterPoint.Builder paramBuilder =
                new ImmutableSvmParameterPoint.Builder();

        // instantiate kernel function
        paramBuilder.kernel = getKernelFunction(
                config.kernelType[0],
                config.degree,
                config.gamma[0],
                config.coef0);

        // set up parameters
        paramBuilder.cache_size = config.cacheSize;
        paramBuilder.eps = config.eps;
        paramBuilder.nu = config.nu;
        paramBuilder.p = config.p;
        paramBuilder.C = config.C[0];
        paramBuilder.shrinking = config.shrinking;
        paramBuilder.probability = config.probability;
        paramBuilder.redistributeUnbalancedC = config.redisC;

        ImmutableSvmParameterPoint param = paramBuilder.build();

        // set default gamma for kernels with gamma in case no gamma was set
        if (param.kernel instanceof GammaKernel
                && ((GammaKernel)param.kernel).getGamma() == 0f) {
            if (config.svmType == SVMConfig.SVMType.EPSILON_SVR
                    || config.svmType == SVMConfig.SVMType.NU_SVR)
                ((GammaKernel)param.kernel).setGamma(1.0f);
            else
                ((GammaKernel)param.kernel).setGamma(0.5f);
        }

        return param;
    }

    private ImmutableSvmParameter setupGridSearchParam(SVMConfig config) {
        ImmutableSvmParameterGrid.Builder paramBuilder =
                new ImmutableSvmParameterGrid.Builder();

        // init kernel set and C set (MUST DO! stupid design...)
        paramBuilder.kernelSet = new HashSet<>();
        paramBuilder.Cset = new HashSet<>();

        // instantiate kernel functions
        for (SVMConfig.KernelType type : config.kernelType)
            for (float gamma : config.gamma) {
                KernelFunction kernel = getKernelFunction(
                        type,
                        config.degree,
                        gamma,
                        config.coef0
                );

                // set default gamma for kernels with gamma
                // in case no gamma was set
                if (kernel instanceof GammaKernel
                        && ((GammaKernel)kernel).getGamma() == 0f) {
                    if (config.svmType == SVMConfig.SVMType.EPSILON_SVR
                            || config.svmType == SVMConfig.SVMType.NU_SVR)
                        ((GammaKernel)kernel).setGamma(1.0f);
                    else
                        ((GammaKernel)kernel).setGamma(0.5f);
                }

                paramBuilder.kernelSet.add(kernel);
            }

        for (float C : config.C)
            paramBuilder.Cset.add(C);

        // set up parameters
        paramBuilder.cache_size = config.cacheSize;
        paramBuilder.eps = config.eps;
        paramBuilder.nu = config.nu;
        paramBuilder.p = config.p;
        paramBuilder.shrinking = config.shrinking;
        paramBuilder.probability = config.probability;
        paramBuilder.redistributeUnbalancedC = config.redisC;


        return paramBuilder.build();
    }


    private SVM getSVM(SVMConfig.SVMType type) {
        switch (type) {
            case C_SVC:
                return new C_SVC();
            case NU_SVC:
                return new Nu_SVC();
            case ONE_CLASS:
                return new OneClassSVC();
            case EPSILON_SVR:
                return new EpsilonSVR();
            case NU_SVR:
                return new Nu_SVR();
            default:
                LOG.error("Not supported SVM type {}", type);
                break;
        }

        return null;
    }

    private KernelFunction getKernelFunction(SVMConfig.KernelType type,
                                             int degree, float gamma, float coef0) {
        switch (type) {
            case LINEAR:
                return new LinearKernel();
            case POLY:
                return new PolynomialKernel(degree, gamma, coef0);
            case RBF:
                return new GaussianRBFKernel(gamma);
            case SIGMOID:
                return new SigmoidKernel(gamma, coef0);
            case PRECOMPUTED:
                return new PrecomputedKernel();
            default:
                LOG.error("Not supported kernel type {}", type);
                break;
        }

        return null;
    }

    private MutableSvmProblem getProblem(int numOfClass) {
        MutableSvmProblem problem;

        int sampleSize = TRAIN.getFeatureMatrix().rows();

        // create problem accordingly
        switch (numOfClass) {
            case 0: // regression
                problem = new MutableRegressionProblemImpl<>(sampleSize);
                break;
            case 1:
                // TODO: not sure if this is correct way
                problem = new MutableRegressionProblemImpl<>(sampleSize);
                break;
            case 2:
                problem = new MutableBinaryClassificationProblemImpl<>(Byte.class, sampleSize);
                break;
            default:
                if (numOfClass > 2)
                    problem = new MutableMultiClassProblemImpl<>(
                            Byte.class,
                            new ByteLabelInverter(),
                            sampleSize,
                            new NoopScalingModel<>()
                    );
                else {
                    LOG.error("Not supported number of classes {}", numOfClass);
                    problem = null;
                }
                break;
        }

        // fill with training data
        INDArray labels = TRAIN.getLabels();
        for (int i = 0; i < sampleSize; ++i) {
            // get sample data
            INDArray row = TRAIN.getFeatureMatrix().getRow(i);

            // fill into sparse vector
            int numOfFeatures = row.columns();
            SparseVector vector = new SparseVector(numOfFeatures);
            for (int v = 0; v < numOfFeatures; ++v) {
                vector.indexes[v] = v + 1;
                vector.values[v] = (float)row.getDouble(v);
            }

            // add into problem
            if (numOfClass == 0 || numOfClass == 1) // regression and one class
                problem.addExampleFloat(vector, (float)labels.getDouble(i));
            else {
                problem.addExample(vector, (byte)DataLoader.roundUp(labels.getDouble(i)));
            }
        }

        return problem;
    }



}
