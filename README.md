# A.I. for Air Quality: Deciphering Benzene Patterns in Urban Environments through Machine Learning

This project presents an analysis of air quality data to address pressing environmental and public health concerns related to air pollution with a focus on Benzene.  Machine learning techniques, such as linear regressions, support vector machines (SVM), and decision trees, were applied to predict and classify Benzene levels as "Good," "Moderate," or "Unhealthy" based on time-stamped concentrations.  The results revealed distinct temporal patterns in pollution levels, allowing the potential to identify peak pollution times and optimal periods for outdoor activities.  

***

## Introduction 

In the era of rapid urbanization and industrial growth, air quality has emerged as a critical environmental and public health concern. Among the various pollutants that deteriorate air quality, Benzene, a volatile organic compound, stands out due to its significant health implications. Classified as a carcinogen, Benzene is predominantly released from industrial emissions and vehicular exhaust, making its presence in urban atmospheres a subject of grave concern. Elevated levels of Benzene are linked to various health issues, including an increased risk of leukemia and other blood disorders, making the monitoring and analysis of its concentration in urban air extremely important.

To solve this problem, supervised machine learning was employed to investigate the Benzene levels in an air quality dataset centered around a densely populated Italian city.  Linear Regression was used to predict continuous Benzene concentrations,  Support Vector Machines (SVM) were utilized to categorize concentrations into risk levels, and Decision Trees were created to identify peak pollution times and optimal periods for outdoor activities.  These tactics create excellent tactics to monitor Benzene levels thus allowing the minimization of exposure to high concentrations of the dangerous carcinogen and reducing its detrimental long-term health consequences.  

The Linear Regression model successfully predicted Benzene levels, capturing a strong linear relationship between time predictors and the target variable.  It demonstrated high predictive power with low error, making it incredibly reliable.  The SVM was able to accurately predict and categorize pollution levels into "Good," "Moderate," and "Unhealthy" risk groups.  Overall, the model showed high precision and recall, especially for the "Unhealthy" category, which is crucial from a public health perspective.  The Decision Tree provided valuable temporal insight into Benzene level fluctuations.  It reveals certain hours of the day, days of the week, and months were more strongly associated with higher levels of Benzene, which hints at patterns of human activity that are influencing pollution levels such as traffic.  

***

## Data

Here is an overview of the dataset, how it was obtained and the preprocessing steps taken, with some plots!

![](assets/IMG/datapenguin.png){: width="500" }

*Figure 1: Here is a caption for my diagram. This one shows a pengiun [1].*

## Modelling

Here are some more details about the machine learning approach, and why this was deemed appropriate for the dataset. 

The model might involve optimizing some quantity. You can include snippets of code if it is helpful to explain things.

```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_features=4, random_state=0)
clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)
clf.predict([[0, 0, 0, 0]])
```

This is how the method was developed.

## Results

Figure X shows... [description of Figure X].

## Discussion

From Figure X, one can see that... [interpretation of Figure X].

## Conclusion

Here is a brief summary. From this work, the following conclusions can be made:
* first conclusion
* second conclusion

Here is how this work could be developed further in a future project.

## References
