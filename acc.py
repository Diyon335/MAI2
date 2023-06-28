import main
from sklearn.metrics import classification_report

import warnings
warnings.simplefilter("ignore")

actual = []
predicted = []
for i in range(50): 
    ac,pr = main.main()
    actual.append(ac)
    predicted.append(pr) 
    print("Complete Run "+str(i+1))


print(actual,predicted)

# Generate classification report
report = classification_report(actual, predicted)

# Print classification report
print("Classification Report:")
print(report)
