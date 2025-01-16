# Clustering and classification with Spotify dataset

## Overview

Data mining about clustering and multi-label classification with dataset from Spotify. 

## Getting Started

To get started with the project, follow the steps below:

Normally, you can skip step 1 if you have the \.venv folder. If not, please start from step 1.

### 1. Create a Python Virtual Environment

First, create a Python virtual environment to isolate your project dependencies. This helps avoid conflicts with other projects.

```bash
python -m venv .venv
```

### 2. Activate the Virtual Environment
On Windows:
```bash
.venv\Scripts\activate
```
On macOS and Linux:
```bash
source .venv/bin/activate
```

### 3. Install Dependencies
Next, install the required packages by executing the following command:

```bash
pip install -r requirements.txt
```

### 4. Run the Project
Once the dependencies are installed, you can run the main script. Use the following command format to execute the program:

```bash
python main.py
```

### 5. Deactivate the Virtual Environment
When you are finished working in the virtual environment, you can deactivate it by running:

```bash
deactivate
```
Your command prompt will return to its normal state, and you will exit the virtual environment.

## Clustering

Clustering with different method including Kmeans++, hierarchical, DBSCAN, GMM.
Note: using column "genre" as ground truth.

### 1. Kmeans++

### 2. Hierarchical clustering

### 3. DBSCAN

### 4. GMM

### 5. Second clustering with feature selection from classification

## Classification

Multi-label classification with different method including random forest, SVM, linear SVM.
Note: using column "genre" as ground truth.

### 1. Random Forest

### 2. SVM

### 3. Linear SVM

### 4. Feature Importance