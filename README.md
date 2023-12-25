# Bank marketing data analysis

The problem involves predicting the likelihood of customer subscribing to the term deposit in a marketing campaign by a bank. The dataset used for this project is the Bank Marketing Dataset, obtained from the [UCI Machine Learning Repository][https://archive.ics.uci.edu/dataset/222/bank+marketing]. Predicting whether a user subscribes to a term subscription based on the marketing data that the bank has, can give more insights for the bank to fine tune their strategy. This will not only enable the identification of potential clients for the campaign but also helps in optimizing the resources, increasing customer satisfaction, and developing tailored marketing strategies for the banking institution.

### Table of Contents

1. **Data collection**
2. **Data visualization**
3. **Imputation**
4. **Categorical encoding**
5. **Standardization of all the columns**
6. **Train test split**
7. **Oversampling**
8. **Dimensionality reduction using PCA**
9. **Cross validation to get the optimal hyper parameters**
10. **KNN and Decision tree implementation from scratch**


### KNN implementation

The implemented algorithm is the k-Nearest Neighbors (KNN) classifier, designed for predicting whether a customer subscribes to a term deposit offered by a bank. The KNN algorithm calculates the distances of all the data points from the target data point and classifies the k-nearest neighbors based on the distance. This is very straightforward algorithm but with increasing data points, time will increase linearly (Time complexity: O(n)) to find the k-nearest neighbors. To tackle this issue, we will be using approximate nearest neighbour techniques which will help in finding the approximate nearest neighbours in O(log(n)) time. I have implemented two approximate nearest neighbour techniques from scratch; ANNOY and KD Tree.

[ANNOY](images/annoy.png)

1. **Using ANNOY(Approximate Nearest Neighbors Oh Yeah) algorithm**
	- **`Constructing the tree`**
		- Firstly we pick up two random points. We draw a line which is equidistant to the points and perpendicular to the line between the two points
		- We divide the points into two groups; points above the line and points below the line
		- Using these parameters(line equation, points above and points below), we create the root node
		- At the next depth, we repeat the process of choosing two random points from the existing set of points and divide the points into two groups(points above and points below) for both left and right child nodes
		- We continue the above process till we reach a node with points less than the minimum required points
	- **`Getting the approximate nearest neighbours`**
		- We start at the root node of the tree and check if the point is above or below the line equation of that node
		- If its below the line, we go to the left subtree or else we got to the right substree
		- We keep parsing through the tree in the above manner till we reach a leaf node. The points above and points below of the root node will be the approximate nearest neighbours of the new point

[KD-Tree](images/kd_tree.webp)

2. **Using KD-Tree algorithm**
	- **`Constructing the KD tree`**
		- Firstly we will calculate the median of i-th(i = depth) coordinate of all the data points
		- We divide the points into two groups, points above the median, points below the median
		- Using the above parameters(median, points above and points below), we create a root node for the KD tree
		- At the next depth, we repeat the process of calculating median, dividing the points into two groups(points above and points below) for both the left and right child nodes
		- At each depth i, we choose the i-th coordinate of the points for calculating the median. If depth(i) is greater than number of dimensions of the point. We choose depth(i) modulus (number of dimensions)
		- We continue the above process till we reach a node with points less than the minimum required points
	- **`Getting the approximate nearest neighbours`**
		- We start at the root node of the tree and check if the depth(i)-th coordinate of the point is less than or more than compared to the median
		- If its less than the median, we go to the left subtree or else we got to the right substree
		- We keep parsing through the tree in the above manner till we reach a leaf node. The points above and points below of the root node will be the approximate nearest neighbours of the new point

### Decision Tree implementation

The algorithm implemented is Decision Tree Classifier. As the problem statement suggests this is a classification problem which predicts if a customer subscribes to a term deposit of a bank. The Decision Tree algorithm is really helpful in handling the non-linear relationships and capturing complex decision rules. This algorithm resembles an intuitive representation of decision making logic, which makes it simple and adaptable to machine learning applications. The scratch implementation of this algorithm tries to closely follow the optimisation techniques and best practices used in sci-kit learn library for a decision tree classifier.

First the Algorithm tries to find a best feature and a best threshold to split on at the root node and divides the training samples into two sets for left child and right respectively. These nodes will become child nodes of the root node which is the initial state for training, thus forming a tree structure. And the same process is repeated for each node of the tree and the tree keeps expanding until it reaches the stopping conditions. And the stopping conditions that are set to stop expanding the tree are as follows:
	- The tree depth reaches specified max depth
	- All the classes of the samples at a node are the same
	- The number of samples at a leaf node are less than the product of min weight fraction leaf and number of samples of its parent

The algorithm uses the following parameters which help in optimization and better performance:
- **`Criterion`**: The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the Information gain
- **`max_depth`**: This parameter is used to set the maximum depth up to which the decision tree can expand to. And once that depth is reached the model stops further splitting of data
- **`random_state`**: This parameter is used to set the randomness of the estimator. This helps in reproducing the results of the model
- **`class_weight`**: This parameter allows us to assign different weights to classes during training. It is particularly useful when dealing with imbalanced datasets. Adjusting class-weight helps in addressing the bias towards the majority class and can lead to a more balanced and accurate model
- **`min_weight_fraction_leaf`**: This parameter sets the minimum weighted fraction of the total number of samples required to be at a leaf node. It requires a fraction of the sum of weights of samples in a leaf, in controlling the minimum size of a leaf node


### Potential discrepancies

1. **KNN**
	- As the algorithms implemented here are approximate nearest neighbour algorithms, we will not be getting the exact k neighbours which are nearest to the target point. 
	- In case of ANNOY algorithm, the splitting of the points is dependent on the randomly choosen points which may lead to discrepancies. To reduce the dependence, I have used multiple trees to get the aggregate of the nearest neighbours for the ANNOY algorithm. 
	- For the KD tree the order of the choice of dimension along which we calculate the median can impact the tree structure leading to different results.
2. **Decision Tree**
	- The algorithm used in the baseline is the same that has been attempted in the current implementation which is essentially the objective to be held up. The baseline model built by utilizing the DecionTreeClassifier library from Scikit Learn amalgamates the CART algorithm of the decision trees and various optimization techniques, like bootstrap sampling and pruning techniques, with some randomness in choosing the features for best split decision. In the prevailing model designed devoid of the inbuilt libraries, a similar approach consisting of a combination of the above-mentioned methods are implemented.


### References

1. https://www.youtube.com/watch?v=DRbjpuqOsjk
2. https://www.youtube.com/watch?v=Y4ZgLlDfKDg
3. https://scikit-learn.org/stable/modules/tree.html