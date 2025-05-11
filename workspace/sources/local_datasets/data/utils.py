import matplotlib.pyplot as plt


def plot_token_length_distribution(dataset):
    df_train_set = dataset.preprocessed_train_set.dataset
    token_lengths = df_train_set['article'].apply(len)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    # First plot
    ax1.hist(token_lengths, bins=50, color='skyblue', edgecolor=None)
    ax1.set_title('Distribution of Article Lengths (in tokens)')
    ax1.set_xlabel('Number of tokens')
    ax1.set_ylabel('Frequency')

    # Second plot
    colors = ['red', 'green']
    labels = ['Fake News', 'Reliable']

    for label, color, name in zip([0, 1], colors, labels):
        mask = df_train_set['label'] == label
        ax2.hist(token_lengths[mask], bins=50, color=color,
                 alpha=0.5, label=name, edgecolor=None)

    ax2.set_title('Distribution of Article Lengths by Label')
    ax2.set_xlabel('Number of tokens')
    ax2.set_ylabel('Frequency')

    # Add vertical lines for model limits
    for limit, label, color in zip([512, 1024], ['512 tokens', '1024 tokens'], ['red', 'blue']):
        ax1.axvline(x=limit, color=color, linestyle='--', label=label)
        ax2.axvline(x=limit, color=color, linestyle='--', label=label)

    ax1.legend()
    ax2.legend()
    fig.suptitle(f'{dataset.name} Dataset Distribution of Article Token Sizes')

    plt.tight_layout()
    plt.show()
