

# Machine Learning Model Building and Evaluation

This project involves building and evaluating various machine learning models using different datasets and techniques. The primary focus is on Random Forest, Lasso Regression, and Decision Tree Classification. The tasks are outlined below with details on data preprocessing, model building, evaluation, and interpretation.

## Project Overview

### 1. Random Forest with NYSE Dataset

**Objective**: Build and evaluate Random Forest models to predict estimated shares outstanding using the NYSE dataset.

#### Data Preprocessing
- **Dataset**: NYSE dataset available on [Kaggle](https://www.kaggle.com/datasets/dgawlik/nyse).
- **Steps**:
  - Load the dataset.
  - Drop the columns: `Ticker Symbol`, `Period Ending`, `For Year`.
  - Split the dataset into training (70%) and testing (30%) sets.

#### Random Forest Model (Default Parameters)
- **Task**: Build a Random Forest model using default parameters.
- **Evaluation**: Compute Mean Squared Error (MSE) on the test set and explain the results.

#### Random Forest Model (min_samples_split=3)
- **Task**: Build a Random Forest model with `min_samples_split` set to 3.
- **Comparison**: Compare this model's performance with the default parameter model and explain the effects of changing `min_samples_split`.

#### Variable Importance
- **Methods**:
  - **Mean Decrease in Impurity (MDI)**: Measures feature importance based on impurity reduction.
  - **Permutation Feature Importance**: Measures importance by evaluating performance drop when feature values are shuffled.
- **Task**: Compute and compare variable importance using both methods. Explain the calculation and comparison results.

#### Lasso Regression
- **Task**: Build a Lasso Regression model using the same train/test split as above.
- **Comparison**: Compare the performance of Lasso Regression with the Random Forest model and discuss the differences.

### 2. Decision Tree Classification with Breast Cancer Dataset

**Objective**: Build and evaluate a Decision Tree Classifier to predict cancer diagnosis using the Breast Cancer dataset.

#### Data Preprocessing
- **Dataset**: Breast Cancer dataset (available through sklearn or other sources).
- **Steps**:
  - Split the dataset into training (70%) and testing (30%) sets.

#### Decision Tree Classifier
- **Task**: Build a Decision Tree Classifier for diagnosis prediction.
- **Evaluation**: Print and interpret the confusion matrix.

#### Tree Visualization
- **Task**: Visualize the decision tree.
- **Explanation**: Identify and discuss the variables in the tree plot.

#### Tree Pruning (Discussion)
- **Task**: Consider the necessity of pruning the decision tree.
- **Explanation**: Discuss where and why pruning might be beneficial, and outline the criteria or measures used to make pruning decisions.

## Setup and Requirements

### Dependencies
- Python (recommended version: 3.x)
- Required libraries: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ml-models-project.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ml-models-project
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
1. **Random Forest and Lasso Regression**:
   - Run the provided scripts in the `random_forest` and `lasso_regression` directories to perform the tasks outlined above.
2. **Decision Tree Classification**:
   - Execute the scripts in the `decision_tree` directory to build and evaluate the decision tree model.

## Notes
- Ensure that all datasets are correctly loaded and preprocessed as described.
- Review the code comments for detailed explanations of each step and function.
- For additional details on model parameters and configurations, refer to the respective sections in the code.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
