package edu.cpp.iipl.netquery.model;

import java.util.List;

/**
 * Created by xing on 4/22/16.
 */
public class ProcessedData {

    private List<String> query;
    private List<String> title;
    private List<String> description;
    private double relevance;
    private double variance;

    public ProcessedData() {
    }

    public ProcessedData(List<String> query, List<String> title, List<String> description, double relevance, double variance) {
        this.query = query;
        this.title = title;
        this.description = description;
        this.relevance = relevance;
        this.variance = variance;
    }

    public List<String> getQuery() {
        return query;
    }

    public void setQuery(List<String> query) {
        this.query = query;
    }

    public List<String> getTitle() {
        return title;
    }

    public void setTitle(List<String> title) {
        this.title = title;
    }

    public List<String> getDescription() {
        return description;
    }

    public void setDescription(List<String> description) {
        this.description = description;
    }

    public double getRelevance() {
        return relevance;
    }

    public void setRelevance(double relevance) {
        this.relevance = relevance;
    }

    public double getVariance() {
        return variance;
    }

    public void setVariance(double variance) {
        this.variance = variance;
    }
}
