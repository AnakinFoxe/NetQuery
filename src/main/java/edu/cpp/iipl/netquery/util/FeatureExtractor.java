package edu.cpp.iipl.netquery.util;

import edu.cpp.iipl.netquery.Setting;
import edu.cpp.iipl.netquery.model.ProcessedData;
import edu.cpp.iipl.tool.feature.extractor.Count;
import edu.cpp.iipl.tool.feature.extractor.Overlap;
import edu.cpp.iipl.tool.feature.extractor.TfIdf;
import edu.cpp.iipl.util.Word2Vec;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.*;

/**
 * Created by xing on 5/9/16.
 */
public class FeatureExtractor {

    private static final Logger LOG = LoggerFactory.getLogger(FeatureExtractor.class);

    private Word2Vec word2Vec;

    private TfIdf tfIdfTitle;
    private TfIdf tfIdfDescription;

    public List<List<Double>> extractFeatures(List<ProcessedData> allData)
            throws IOException {
        List<List<Double>> features = new ArrayList<>();

        if (allData == null || allData.size() == 0)
            return features;

        // prepare for word2vec feature extraction
        if (Setting.INCLUDE_FEAT_WORD2VEC) {
            word2Vec = new Word2Vec();
            word2Vec.readVectors(Setting.WORD2VEC_VECTORS);
        }

        // prepare for tf-idf feature extraction
        if (Setting.INCLUDE_FEAT_TFIDF)
            prepareTfIdf(allData);

        // extract features
        for (ProcessedData data : allData) {
            List<Double> feature = new ArrayList<>();

            // counting based features
            if (Setting.INCLUDE_FEAT_COUNT)
                addFeatCount(data, feature);

            // overlapping based features
            if (Setting.INCLUDE_FEAT_OVERLAP)
                addFeatOverlap(data, feature);

            // tf-idf based features
            if (Setting.INCLUDE_FEAT_TFIDF)
                addFeatTfIdf(allData.indexOf(data), data, feature);

            // word2vec based features
            if (Setting.INCLUDE_FEAT_WORD2VEC)
                addFeatWord2Vec(data, feature);

            features.add(feature);
        }

        LOG.warn("Feature extraction accomplished.");

        return features;
    }

    private void addFeatCount(ProcessedData data,
                              List<Double> feature) {

        // count of unigrams
        feature.add((double) Count.numOfWords(data.getUnigramQuery()));
        feature.add((double) Count.numOfWords(data.getUnigramTitle()));
        feature.add((double) Count.numOfWords(data.getUnigramDesc()));

        // count of bigrams
        if (Setting.INCLUDE_FEAT_WITH_BIGRAM) {
            feature.add((double) Count.numOfWords(data.getBigramQuery()));
            feature.add((double) Count.numOfWords(data.getBigramTitle()));
            feature.add((double) Count.numOfWords(data.getBigramDesc()));
        }

        // count of trigrams
        if (Setting.INCLUDE_FEAT_WITH_TRIGRAM) {
            feature.add((double) Count.numOfWords(data.getTrigramQuery()));
            feature.add((double) Count.numOfWords(data.getTrigramTitle()));
            feature.add((double) Count.numOfWords(data.getTrigramDesc()));
        }
    }

    private void addNGramFeatOverlap(List<String> query,
                                     List<String> title,
                                     List<String> description,
                                     List<Double> feature) {
        // first word of query in title/description (overlap2)
        int numOfFirstWordQueryInTitle = 0;
        int numOfFirstWordQueryInDescription = 0;
        if (query.size() > 0) {
            String firstWord = query.get(0);
            numOfFirstWordQueryInTitle = Overlap.numOfWordInWords(firstWord, title);
            numOfFirstWordQueryInDescription = Overlap.numOfWordInWords(firstWord, description);
        }

        // last word of query in title/description (overlap3)
        int numOfLastWordQueryInTitle = 0;
        int numOfLastWordQueryInDescription = 0;
        if (query.size() > 0) {
            String lastWord = query.get(query.size() - 1);
            numOfLastWordQueryInTitle = Overlap.numOfWordInWords(lastWord, title);
            numOfLastWordQueryInDescription = Overlap.numOfWordInWords(lastWord, description);
        }

        // all the words of query in title/description (overlap4)
        int numOfWordsQueryInTitle = Overlap.numOfWords1InWords2(query, title);
        int numOfWordsQueryInDescription = Overlap.numOfWords1InWords2(query, description);

        // ratio of (overlap4 / length of title/description)
        double ratioOfTitle = 0;
        double ratioOfDescription = 0;
        if (title.size() > 0)
            ratioOfTitle = (double) numOfWordsQueryInTitle / title.size();
        if (description.size() > 0)
            ratioOfDescription = (double) numOfWordsQueryInDescription / description.size();

        feature.add((double) numOfFirstWordQueryInTitle);
        feature.add((double) numOfFirstWordQueryInDescription);
        feature.add((double) numOfLastWordQueryInTitle);
        feature.add((double) numOfLastWordQueryInDescription);
        feature.add((double) numOfWordsQueryInTitle);
        feature.add((double) numOfWordsQueryInDescription);
        feature.add(ratioOfTitle);
        feature.add(ratioOfDescription);
    }

    private void addFeatOverlap(ProcessedData data,
                                List<Double> feature) {

        // complete query in title/description (overlap1)
        int numOfQueryInTitle = Overlap.numOfStr1InStr2(data.getUnigramQuery(), data.getUnigramTitle());
        int numOfQueryInDescription = Overlap.numOfStr1InStr2(data.getUnigramQuery(), data.getUnigramDesc());

        feature.add((double) numOfQueryInTitle);
        feature.add((double) numOfQueryInDescription);

        // unigram
        addNGramFeatOverlap(
                data.getUnigramQuery(),
                data.getUnigramTitle(),
                data.getUnigramDesc(),
                feature);

        // bigram
        if (Setting.INCLUDE_FEAT_WITH_BIGRAM)
            addNGramFeatOverlap(
                    data.getBigramQuery(),
                    data.getBigramTitle(),
                    data.getBigramDesc(),
                    feature);

        // trigram
        if (Setting.INCLUDE_FEAT_WITH_TRIGRAM)
            addNGramFeatOverlap(
                    data.getTrigramQuery(),
                    data.getTrigramTitle(),
                    data.getTrigramDesc(),
                    feature);
    }

    // Note: only use unigram for tf-idf
    // because we apply Bag-of-Word concept
    // and not worth it for bigram and trigram
    private void prepareTfIdf(List<ProcessedData> allData) {
        List<List<String>> titles = new ArrayList<>();
        List<List<String>> descriptions = new ArrayList<>();
        Set<String> dict = new HashSet<>();
        for (ProcessedData data : allData) {
            // update dictionary for tf-idf computation
            dict.addAll(data.getUnigramQuery());
//            dict.addAll(data.getUnigramTitle());
//            dict.addAll(data.getUnigramDesc());

            // prepare documents
            titles.add(data.getUnigramTitle());
            descriptions.add(data.getUnigramDesc());
        }
        // compute tf & tf-idf
        tfIdfTitle = new TfIdf(dict, titles);
        tfIdfDescription = new TfIdf(dict, descriptions);
    }


    private void addFeatTfIdf(int docId,
                              ProcessedData data,
                              List<Double> feature) {

        if (tfIdfTitle == null || tfIdfDescription == null) {
            LOG.error("TF-IDF computation was not prepared");
            return;
        }

        List<String> query = data.getUnigramQuery();
//        List<String> title = data.getUnigramTitle();

        // cosine similarity of query vs title
        double cosineQnT = tfIdfTitle.cosineSimilarity(docId, query);

        // cosine similarity of query vs description
        double cosineQnD = tfIdfDescription.cosineSimilarity(docId, query);

        // cosine similarity of title vs description
//        double cosineTnD = tfIdfDescription.cosineSimilarity(docId, title);


        feature.add(cosineQnT);
        feature.add(cosineQnD);
//        feature.add(cosineTnD);
    }


    private Double[] averageVec(List<String> tokens) {
        Double[] averaged = new Double[word2Vec.getNumOfVectors()];
        for (int i = 0; i < averaged.length; ++i)
            averaged[i] = 0.0;


        int numOfVec = 0;
        for (String token : tokens) {
            double[] vec = word2Vec.getVectors(token);

            if (vec != null) {
                for (int i = 0; i < averaged.length; ++i) {
                    averaged[i] += vec[i];
                    ++numOfVec;
                }
            }
        }

        if (numOfVec != 0)
            for (int i = 0; i < averaged.length; ++i)
                averaged[i] /= numOfVec;

        return averaged;
    }

    private double cosine(Double[] vec1, Double[] vec2) {
        double cosine = 0;

        for (int i = 0; i < vec1.length; ++i)
            cosine += vec1[i] * vec2[i];

        return cosine;
    }

    private void addFeatWord2Vec(ProcessedData data, List<Double> feature) {
        List<String> query = data.getUnigramQuery();
        List<String> title = data.getUnigramTitle();
        List<String> description = data.getUnigramDesc();

        // average vec for query, title & description
        Double[] averagedQuery = averageVec(query);
        Double[] averagedTitle = averageVec(title);
        Double[] averagedDesc = averageVec(description);

        // cosine similarity of query vs title based on word2vec
        double cosineQvT = cosine(averagedQuery, averagedTitle);

        // cosine similarity of query vs description based on word2vec
        double cosineQvD = cosine(averagedQuery, averagedDesc);

        // cosine similarity of title vs description based on word2vec
        double cosineTvD = cosine(averagedTitle, averagedDesc);

        feature.addAll(Arrays.asList(averagedQuery));
        feature.addAll(Arrays.asList(averagedTitle));
        feature.addAll(Arrays.asList(averagedDesc));
        feature.add(cosineQvT);
        feature.add(cosineQvD);
        feature.add(cosineTvD);
    }
}
