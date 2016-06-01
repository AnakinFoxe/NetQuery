package edu.cpp.iipl.netquery.svm;

/**
 * Created by xing on 5/30/16.
 */
public class SVMConfig {

    public enum SVMType {
        C_SVC,              // C-Support Vector Classification
        NU_SVC,             // Nu-Support Vector Classification
        ONE_CLASS,          // for Distribution Estimation
        EPSILON_SVR,        // epsilon-Support Vector Regression
        NU_SVR              // Nu-Support Vector Regression
    }

    public enum KernelType {
        LINEAR,
        POLY,
        RBF,
        SIGMOID,
        PRECOMPUTED         // not recommended for our work
    }

    public int numOfClass           = 0;        // 0 for regression, 1-n for classification

    public SVMType svmType          = SVMType.C_SVC;

    // kernel configs
    public KernelType[] kernelType  =
            {
                    SVMConfig.KernelType.LINEAR,
                    SVMConfig.KernelType.POLY,
                    SVMConfig.KernelType.RBF,
                    SVMConfig.KernelType.SIGMOID
            };
    public int degree               = 3;
    public float[] gamma            =
            {
                    1.0f / (1<<13),
                    1.0f / (1<<11),
                    1.0f / (1<<9),
                    1.0f / (1<<7),
                    1.0f / (1<<5),
                    1.0f / (1<<3),
                    0.5f,
                    1.0f,
                    2.0f,
                    1<<3
            };
    public float coef0              = 0;

    // other configs (default value)
    public float cacheSize          = 400;      // in MB
    public float eps                = 1e-3f;    // stopping criteria
    public float nu                 = 0.5f;     // for NU_SVC, NU_SVR, and ONE_CLASS
    public float p                  = 0.1f;     // for EPSILON_SVR
    public float[] C                =           // for C_SVC, EPSILON_SVR and NU_SVR
            {
                    1.0f / (1<<5),
                    1.0f / (1<<3),
                    0.5f,
                    1f,
                    1<<3,
                    1<<5,
                    1<<7,
                    1<<9,
                    1<<11,
                    1<<13

            };
    public boolean shrinking        = true;     // use the shrinking heuristics or not
    public boolean probability      = false;    // do probability estimates or not
    public boolean redisC           = true;     // For unbalanced data, redistribute the misclassification cost C
                                                // according to the numbers of examples in each class, so that each
                                                // class has the same total misclassification weight assigned to it
                                                // and the average is param.C
}
