package edu.cpp.iipl.deepquery.util;

import edu.cpp.iipl.deepquery.Setting;
import edu.cpp.iipl.deepquery.model.Data;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
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
        }

        LOG.info("Data set loading accomplished.");

    }
}
