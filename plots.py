import matplotlib.pyplot as plt
import seaborn as sns

def generate_histogram_plot(accepted_data, rejected_data, x, bins, kde, label_accepted, label_rejected, palette, hue, rot, xlabel, ylabel, title, save=True):
    plt.figure(figsize=(6,4))
    sns.histplot(data=accepted_data, x=x, bins=bins, kde=kde, label=label_accepted, palette=palette, hue=hue)
    sns.histplot(data=rejected_data, x=x, bins=bins, kde=kde, label=label_rejected, palette=palette, hue =hue)
    plt.xticks(rotation=rot, fontsize=9)
    plt.xlabel(xlabel, fontsize=9)
    plt.ylabel(ylabel, fontsize=9)
    plt.title(title, fontsize=9)
    plt.legend(fontsize=9)
    plt.tight_layout()
    if save:
        filename = title.replace(" ", "_") + ".png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def plot_channelid_distribution(data, save=True):
    accepted_counts = data[data['HAS_REAL_ORDER'] == '1']['CHANNELID'].value_counts()

    # Sort CHANNELID categories based on accepted_counts
    order = accepted_counts.index
    title = 'Distribution of Acceptance and Rejection by CHANNELID'
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=data, x='CHANNELID', hue='HAS_REAL_ORDER', palette='colorblind', order=order)
    plt.xlabel('CHANNELID', fontsize=9)
    plt.ylabel('Count', fontsize=9)
    plt.title(title, fontsize=9)
    plt.xticks(rotation=90, fontsize=9)
    # Calculate and annotate the percentages
    total_counts = len(data)
    min_percentage_to_display = 0.5  
    for p in ax.patches:
        height = p.get_height()
        percentage = height / total_counts * 100

        # Display only if the percentage is higher than the threshold
        if percentage > min_percentage_to_display:
            ax.annotate(f'{percentage:.1f}%', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', rotation=90)

    custom_labels = ['rejected','accepted']
    plt.legend(labels=custom_labels, fontsize=9)
    plt.ylim(0, 30000)
    if save:
        filename = title.replace(" ", "_") + ".png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.tight_layout()  
    plt.show()