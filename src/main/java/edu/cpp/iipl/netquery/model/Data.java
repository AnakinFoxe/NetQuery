package edu.cpp.iipl.netquery.model;

/**
 * Created by xing on 4/22/16.
 */
public class Data {

    private String query;
    private String title;
    private String description;
    private double relevance;
    private double variance;

    public Data() {
    }

    public Data(String query, String title, String description, double relevance, double variance) {
        this.query = query;
        this.title = title;
        this.description = description;
        this.relevance = relevance;
        this.variance = variance;
    }

    public String getQuery() {
        return query;
    }

    public void setQuery(String query) {
        this.query = query;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
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
