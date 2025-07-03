# Website Traffic Clustering Project

## Overview

This project performs an unsupervised machine learning analysis on website traffic data using **K-Means Clustering**. The primary goal is to identify patterns or clusters in the relationship between **Search Volume** and **Traffic Cost**.

The project is implemented using **Python**, with the analysis conducted in a **Jupyter Notebook**.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Data Loading & Exploration](#data-loading--exploration)
3. [Data Visualization](#data-visualization)
4. [K-Means Clustering](#k-means-clustering)
5. [Elbow Method for Optimal Clusters](#elbow-method-for-optimal-clusters)
6. [Cluster Visualization](#cluster-visualization)
7. [Conclusion](#conclusion)
8. [Technologies & Libraries Used](#technologies--libraries-used)
9. [Tools](#tools)
10. [How to Run](#how-to-run)

---

## Introduction

This notebook focuses on analyzing website traffic data to understand the relationship between:

* **Search Volume**: The number of times a keyword is searched.
* **Traffic Cost**: The estimated cost of generating traffic for the keyword.

The aim is to group the data into clusters based on these two features.

---

## Data Loading & Exploration

* The dataset (`website_traffic_data.csv`) is loaded using **Pandas**.
* Basic exploratory data analysis (`head()`, `info()`, `describe()`) is performed to understand the structure and summary statistics of the data.

---

## Data Visualization

* A **scatter plot** is created using **Seaborn** and **Matplotlib** to visualize the relationship between **Search Volume** and **Traffic Cost**.
* This helps in visually inspecting possible natural clusters.

---

## K-Means Clustering

* The two key features (**Search Volume** and **Traffic Cost**) are selected and converted into a 2D array for clustering.
* **K-Means clustering** from **scikit-learn** is applied to group the data.

---

## Elbow Method for Optimal Clusters

* The **Elbow Method** is used to find the optimal number of clusters by plotting clustering inertia scores for cluster counts from 1 to 10.
* The "elbow point" suggests that **2 clusters** is a good choice.

---

## Cluster Visualization

* K-Means clustering is run with **2 clusters**.
* Cluster labels are added to the dataset.
* A scatter plot is created to visualize the clusters:

  * **Cluster 0**: Blue
  * **Cluster 1**: Green
  * **Cluster Centroids**: Red stars

---

## Conclusion

* The clustering reveals distinct groupings in the data based on **Search Volume** and **Traffic Cost**.
* Visualizations help in understanding the separation and central points of the clusters.

---

## Technologies & Libraries Used

* Python (Jupyter Notebook)
* Pandas
* NumPy
* Matplotlib
* Seaborn
* scikit-learn

---

## Tools

* **Jupyter Notebook**: Interactive analysis and visualization.
* **Visual Studio Code** or other Python IDEs: Development environment.
* **Git & GitHub**: Version control and project sharing.
* **Command Line Interface (CLI)**: For installing dependencies and running notebooks.

---

## How to Run

1. Clone this repository.
2. Open the `.ipynb` file in **Jupyter Notebook**.
3. Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

4. Run the notebook cells step-by-step.

---
