package edu.cpp.iipl.netquery.util;

import edu.cpp.iipl.netquery.Setting;
import edu.cpp.iipl.netquery.model.Data;
import edu.cpp.iipl.netquery.model.ProcessedData;
import edu.cpp.iipl.tool.preprocessor.SpellCorrector;
import edu.cpp.iipl.tool.preprocessor.Stemmer;
import edu.cpp.iipl.tool.preprocessor.Stopword;
import edu.cpp.iipl.tool.preprocessor.TextFormatter;
import edu.cpp.iipl.util.NGram;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;

/**
 * Created by xing on 5/9/16.
 */
public class Preprocessor {

    private static final Logger LOG = LoggerFactory.getLogger(Preprocessor.class);


    public static List<ProcessedData> preprocess(List<Data> allData) {
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

            // stemming
            procQuery = st.stemWords(procQuery);
            procTitle = st.stemWords(procTitle);
            procDescription = st.stemWords(procDescription);

            // create bigrams from unigrams
            List<String> bigramQuery = NGram.unigram2NGram(2, "<s>", procQuery);
            List<String> bigramTitle = NGram.unigram2NGram(2, "<s>", procTitle);
            List<String> bigramDesc = NGram.unigram2NGram(2, "<s>", procDescription);

            // create trigrams from unigrams
            List<String> trigramQuery = NGram.unigram2NGram(3, "<s>", procQuery);
            List<String> trigramTitle = NGram.unigram2NGram(3, "<s>", procTitle);
            List<String> trigramDesc = NGram.unigram2NGram(3, "<s>", procDescription);

            // remove stopwords
            // only for unigrams
            List<String> unigramQuery = sw.rmStopword(procQuery);
            List<String> unigramTitle = sw.rmStopword(procTitle);
            List<String> unigramDesc = sw.rmStopword(procDescription);


            processedDataList.add(new ProcessedData(
                    unigramQuery, unigramTitle, unigramDesc,
                    bigramQuery, bigramTitle, bigramDesc,
                    trigramQuery, trigramTitle, trigramDesc,
                    data.getRelevance(), data.getVariance()));
        }

        LOG.warn("Preprocessing accomplished.");

        return processedDataList;
    }
}
