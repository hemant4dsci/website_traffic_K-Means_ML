# 📊 K-Means Clustering – Website Traffic Segmentation

This project applies **K-Means Clustering** (unsupervised machine learning) to segment website traffic data into meaningful groups based on metrics like search volume, traffic, and traffic cost.  
The goal is to identify distinct patterns in web traffic behavior for deeper marketing and SEO insights.

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Tools & Technologies](#-tools--technologies)
3. [Dataset](#-dataset)
4. [Workflow](#-workflow)
5. [Installation & Usage](#-installation--usage)
6. [Results & Insights](#-results--insights)
7. [Tech Stack](#-tech-stack)
8. [License](#-license)
9. [Contributing](#-contributing)
10. [Author](#-author)

---

## 🚀 Project Overview

The notebook `k_means_cluster_web_traffic.ipynb` demonstrates the complete **data science pipeline** for clustering, including:
- Data exploration
- Feature correlation analysis
- Data preprocessing
- Optimal cluster selection using the Elbow Method
- K-Means clustering
- PCA-based 2D visualization of results

---

## 🛠 Technologies & Libraries Used

| Technology                       | Description                                 |
| -------------------------------- | ------------------------------------------- |
| 🐍 **Python**            | Core programming language for analysis and modeling |
| 📓 **Jupyter Notebook**  | Interactive coding and documentation |
| 📊 **pandas**            | Data manipulation and analysis |
| 🔢 **numpy**             | Numerical computations |
| 🤖 **scikit-learn**      | Machine learning library (KMeans, StandardScaler, PCA) |
| 📈 **Seaborn**           | Statistical data visualization |
| 📉 **Matplotlib**        | Plotting and charting |

---

## 📂 Dataset

**File:** `website_traffic_data.csv`  
Contains website keyword metrics including:
- **Search Volume**
- **Traffic**
- **Traffic (%)**
- **Traffic Cost**
- **Traffic Cost (%)**

> Sensitive or irrelevant columns (like IDs) are removed before clustering.

---

## 🛠 Workflow

1. **Data Loading & Inspection**
   - Load dataset using `pandas`
   - Check structure with `.info()` and `.describe()`

2. **Correlation Analysis**
   - Generate a **heatmap** of numeric feature correlations
   - Observations:
     - Search Volume strongly correlates (>0.70) with Traffic, Traffic (%), Traffic Cost, and Traffic Cost (%)
     - Some features show perfect correlation (1.00)
``python
plt.figure(figsize=(10,7))
sns.heatmap(data=df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.show()
``

3. **Data Preprocessing**
   - Keep only relevant numeric columns
   - Fill missing values with the **median**
   - Standardize features with `StandardScaler`

4. **Choosing Optimal Clusters**
   - Use **Elbow Method** (inertia vs. k) to find the elbow point

5. **Applying K-Means**
   - Train K-Means with the chosen `k`
   - Assign **Cluster Labels** back to the DataFrame

6. **Dimensionality Reduction with PCA**
   - Reduce from n-dimensional space to **2D**
   - Transform **centroids** into PCA space for plotting

---

## ▶️ How to Run

1. Clone this repository.
2. Open the `.ipynb` file in **Jupyter Notebook**.
3. Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

4. Run the notebook cells step-by-step.

---

## ✅ Conclusion

* The clustering reveals distinct groupings in the data based on **Search Volume** and **Traffic Cost**.
* Visualizations help in understanding the separation and central points of the clusters.

---

## 👤 About Me

Hi, I'm Hemant, a data enthusiast passionate about turning raw data into meaningful business insights.

📫 **Let’s connect:**
- LinkedIn : [LinkedIn Profile](https://www.linkedin.com/in/hemant1491/)  
- Email : hemant4dsci@gmail.com

---
