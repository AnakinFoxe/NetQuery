package edu.cpp.iipl.netquery.model;

import java.util.List;

/**
 * Created by xing on 4/22/16.
 */
public class ProcessedData {

    // unigram form
    private List<String> unigramQuery;
    private List<String> unigramTitle;
    private List<String> unigramDesc;

    // bigram form
    private List<String> bigramQuery;
    private List<String> bigramTitle;
    private List<String> bigramDesc;

    // trigram form
    private List<String> trigramQuery;
    private List<String> trigramTitle;
    private List<String> trigramDesc;


    private double relevance;
    private double variance;

    public ProcessedData() {
    }

    public ProcessedData(List<String> unigramQuery, List<String> unigramTitle, List<String> unigramDesc,
                         List<String> bigramQuery, List<String> bigramTitle, List<String> bigramDesc,
                         List<String> trigramQuery, List<String> trigramTitle, List<String> trigramDesc,
                         double relevance, double variance) {
        this.unigramQuery = unigramQuery;
        this.unigramTitle = unigramTitle;
        this.unigramDesc = unigramDesc;
        this.bigramQuery = bigramQuery;
        this.bigramTitle = bigramTitle;
        this.bigramDesc = bigramDesc;
        this.trigramQuery = trigramQuery;
        this.trigramTitle = trigramTitle;
        this.trigramDesc = trigramDesc;
        this.relevance = relevance;
        this.variance = variance;
    }

    public List<String> getUnigramQuery() {
        return unigramQuery;
    }

    public void setUnigramQuery(List<String> unigramQuery) {
        this.unigramQuery = unigramQuery;
    }

    public List<String> getUnigramTitle() {
        return unigramTitle;
    }

    public void setUnigramTitle(List<String> unigramTitle) {
        this.unigramTitle = unigramTitle;
    }

    public List<String> getUnigramDesc() {
        return unigramDesc;
    }

    public void setUnigramDesc(List<String> unigramDesc) {
        this.unigramDesc = unigramDesc;
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

    public List<String> getBigramQuery() {
        return bigramQuery;
    }

    public void setBigramQuery(List<String> bigramQuery) {
        this.bigramQuery = bigramQuery;
    }

    public List<String> getBigramTitle() {
        return bigramTitle;
    }

    public void setBigramTitle(List<String> bigramTitle) {
        this.bigramTitle = bigramTitle;
    }

    public List<String> getBigramDesc() {
        return bigramDesc;
    }

    public void setBigramDesc(List<String> bigramDesc) {
        this.bigramDesc = bigramDesc;
    }

    public List<String> getTrigramQuery() {
        return trigramQuery;
    }

    public void setTrigramQuery(List<String> trigramQuery) {
        this.trigramQuery = trigramQuery;
    }

    public List<String> getTrigramTitle() {
        return trigramTitle;
    }

    public void setTrigramTitle(List<String> trigramTitle) {
        this.trigramTitle = trigramTitle;
    }

    public List<String> getTrigramDesc() {
        return trigramDesc;
    }

    public void setTrigramDesc(List<String> trigramDesc) {
        this.trigramDesc = trigramDesc;
    }
}
