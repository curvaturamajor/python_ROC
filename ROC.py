from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt


actual_value = [1,1,0,1,0,0,0,1,0,1,1,1]
predictions = [0.98,0.43,0.3,0.55,0.35,0.2,0.1,0.60,0.03,0.85,0.35,0.58]
prediction_2 = [0.63,0.54,0.4,0.7,0.4,0.66,0.51,0.4,0.3,0.8,0.71,0.4]

false_positive_rate, sensitivity, threshold_values = roc_curve(actual_value,predictions)
auc_1 = auc(false_positive_rate,sensitivity)

false_positive_rate_2, sensitivity_2, threshold_values_2 = roc_curve(actual_value,predictions_2)
auc_2 = auc(false_positive_rate_2,sensitivity_2)


plt.plot(false_positive_rate,sensitivity,label="Model 1 AUC = %0.3f" %auc_1)
plt.plot(false_positive_rate_2,sensitivity_2,label="Model 2 AUC = %0.3f" %auc_2)

plt.xlabel("False Positive Rate")
plt.ylabel("Sensitivity")

plt.legend(loc='lower right')

plt.show()
