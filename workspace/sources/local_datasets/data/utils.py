import matplotlib.pyplot as plt
from ..dataset import Dataset


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


def plot_label_distribution(data, dataset_name):
    label_mapping = Dataset.LABELS_MAPPING
    label_counts = data['label'].value_counts().sort_index()  # Sort by index to match tick order
    plt.figure(figsize=(8, 5))  # Increased figure height
    plt.bar(label_counts.index, label_counts.values, color=['blue', 'red'], alpha=0.6)
    plt.xticks([0, 1], [label_mapping[0], label_mapping[1]])
    plt.title(f'Label Balance in {dataset_name} Dataset', pad=20, fontsize=16)
    plt.xlabel('Label', fontweight='bold', fontsize=12)
    plt.ylabel('Count', fontweight='bold', fontsize=12)

    # Calculate total samples
    total = label_counts.sum()

    # Add numbers and ratios on top of the bars with larger font and bold text
    for i, value in enumerate(label_counts.values):
        ratio = value / total * 100
        plt.text(i, value + 15, f"{value}\n({ratio:.1f}%)",
                 ha='center', fontsize=10, fontweight='bold')

    plt.ylim(0, max(label_counts.values) * 1.15)  # Add 15% margin on top
    plt.show()
    plt.savefig('images/label_distribution.png')
    plt.close()


def plot_article_length_distribution(df, dataset_name, save_path='images/article_lengths_distribution.png'):
    # Font sizes
    title_font_size = 14
    label_font_size = 12
    super_title_font_size = 16

    # Calculate article lengths
    article_lengths = df['article'].str.len()

    # Create figure with 4 subplots in 2x2 grid
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # First plot - full distribution
    ax1.hist(article_lengths, bins=50, color='skyblue', edgecolor='skyblue', linewidth=0)
    ax1.set_title('Compound Distribution', fontweight='bold', fontsize=title_font_size)
    ax1.set_xlabel('Number of characters', fontsize=label_font_size)
    ax1.set_ylabel('Frequency', fontsize=label_font_size)

    # Second plot - distribution by label (overlapped)
    colors = ['red', 'green']
    labels = ['Fake News', 'Reliable']

    for label, color, name in zip([0, 1], colors, labels):
        mask = df['label'] == label
        ax2.hist(article_lengths[mask], bins=50, color=color,
                 alpha=0.5, label=name, edgecolor=color, linewidth=0)

    ax2.set_title('Distribution by Label\n(overlapped)', fontweight='bold', fontsize=title_font_size)
    ax2.set_xlabel('Number of characters', fontsize=label_font_size)
    ax2.set_ylabel('Frequency', fontsize=label_font_size)
    ax2.legend()

    # Third plot - Fake News distribution
    mask_fake = df['label'] == 0
    ax3.hist(article_lengths[mask_fake], bins=50, color='red', alpha=0.5, edgecolor='red', linewidth=0)
    ax3.set_title('Fake News Distribution', fontweight='bold', fontsize=title_font_size)
    ax3.set_xlabel('Number of characters', fontsize=label_font_size)
    ax3.set_ylabel('Frequency', fontsize=label_font_size)

    # Fourth plot - Reliable News distribution
    mask_reliable = df['label'] == 1
    ax4.hist(article_lengths[mask_reliable], bins=50, color='green', alpha=0.5, edgecolor='green', linewidth=0)
    ax4.set_title('Reliable News Distribution', fontweight='bold', fontsize=title_font_size)
    ax4.set_xlabel('Number of characters', fontsize=label_font_size)
    ax4.set_ylabel('Frequency', fontsize=label_font_size)

    fig.suptitle(f'{dataset_name} Dataset Distribution of Article Lengths (in characters)',
                 fontweight='bold', fontsize=super_title_font_size, y=1.01)
    plt.tight_layout()

    # Save plots
    plt.savefig(save_path)
    plt.show()
