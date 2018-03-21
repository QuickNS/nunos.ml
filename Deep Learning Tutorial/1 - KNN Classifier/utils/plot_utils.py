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
    ax.set_xlabel('Predicted', labelpad=20)
    ax.set_ylabel('Truth', labelpad=20)

    xtick_labels = list()
    ytick_labels = list()
    xtick_totals = np.sum(confusionMatrix, axis=0)
    ytick_totals = np.sum(confusionMatrix, axis=1)

    for i in range(len(labels)):
        xtick_labels.append("{0}\nTotal: {1}".format(labels[i], xtick_totals[i]))
        ytick_labels.append("{0}\nTotal: {1}".format(labels[i], ytick_totals[i]))
        
    ax.set_xticklabels(xtick_labels, rotation='horizontal', ha='center', size=12)
    ax.set_yticklabels(ytick_labels, rotation='horizontal', va='center', size=12)
    
    plt.show()

def plotClassificationReport(classification_report, title='Classification report', cmap='RdBu'):
    
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    lines = classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2 : (len(lines) - 2)]:
        t = line.strip().split()
        if len(t) < 2: continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        plotMat.append(v)

    t = lines[-2].strip().split()
    scores = t[3:-1]
    total_samples = int(t[-1])
   
        
    xlabel = 'Metrics'
    ylabel = 'Classes'
    metrics = ['Precision', 'Recall', 'F1-score']
    xticklabels = ['{0}\n({1})'.format(metrics[idx], score) for idx, score  in enumerate(scores)]
    yticklabels = ['{0}\n({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    
    f, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(np.array(plotMat), annot=True,annot_kws={"size": 12} ,linewidths=2, fmt=".2f", vmin=0, vmax=1, cmap=cmap, ax=ax)
    ax.set_title("%s\nTotal: %d" % (title, total_samples))
    ax.set_xlabel(xlabel,labelpad=20)
    ax.set_ylabel(ylabel, labelpad=20)
    ax.set_xticklabels(xticklabels, rotation='horizontal', ha='center')
    ax.set_yticklabels(yticklabels, rotation='horizontal', va='center')
   
