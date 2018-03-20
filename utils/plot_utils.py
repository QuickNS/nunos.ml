def showLabelDistributionByCategory(data, categorical_feature, label):
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(121)
    sns.boxplot(x=categorical_feature,y=label, data=data,ax=ax)
    plt.xticks(rotation=90)
    ax2 = fig.add_subplot(122)
    sns.pointplot(x=categorical_feature,y=label, data=data,ax=ax2)
    plt.xticks(rotation=90)
    plt.show()

def showCorrMatrix(matrix):   
    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(matrix, annot=True, ax=ax, center=0)
    tick_marks = [i for i in range(len(matrix.columns))]
    ax.set_xticklabels(matrix.columns)
    ax.set_yticklabels(matrix.columns)

def plotConfusionMatrix(confusionMatrix, labels):
        
    #plotting the confusion matrix
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(confusionMatrix, annot=True,annot_kws={"size": 12} ,linewidths=2, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Truth')

    xtick_labels = list()
    ytick_labels = list()
    xtick_totals = np.sum(confusionMatrix, axis=0)
    ytick_totals = np.sum(confusionMatrix, axis=1)

    print (xtick_totals)
    print (ytick_totals)

    for i in range(len(labels)):
        xtick_labels.append("{0}\nTotal: {1}".format(labels[i], xtick_totals[i]))
        ytick_labels.append("{0}\nTotal: {1}".format(labels[i], ytick_totals[i]))
        
    ax.set_xticklabels(xtick_labels, rotation='horizontal', ha='center', size=12)
    ax.set_yticklabels(ytick_labels, rotation='horizontal', va='center', size=12)
    
    plt.show()