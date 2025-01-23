import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read the data
data = pd.read_csv('../Files/combined.csv')

# Set consistent colors
colors = {
    1: 'orange',
    2: 'blue',
    3: 'green',
    4: 'red',
    5: 'purple',
    6: 'brown',
    7: 'pink'
}

# 1. Bar Plot
plt.figure(figsize=(12, 6))
action_counts = data['action'].value_counts().sort_index()
bars = plt.bar(action_counts.index, action_counts.values)
for idx, bar in enumerate(bars):
    bar.set_color(colors[idx + 1])
plt.title('Distribution of Actions', fontsize=14, pad=20)
plt.xlabel('Action Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 2. Main Scatter Plot
plt.figure(figsize=(12, 8))
for action in colors:
    mask = data['action'] == action
    plt.scatter(data[mask]['queue1'],
               data[mask]['queue2'],
               label=f'Action {action}',
               alpha=0.6,
               c=colors[action],
               s=100)
plt.title('Queue1 vs Queue2 by Action Type', fontsize=14, pad=20)
plt.xlabel('Queue 1', fontsize=12)
plt.ylabel('Queue 2', fontsize=12)
plt.legend(title='Action Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 3. Box Plots - Queue1
plt.figure(figsize=(12, 6))
box_colors = [colors[action] for action in sorted(colors.keys())]
bp1 = plt.boxplot([data[data['action'] == action]['queue1'] for action in sorted(colors.keys())],
                  labels=sorted(colors.keys()),
                  patch_artist=True)
for patch, color in zip(bp1['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
plt.title('Queue1 Distribution by Action', fontsize=14, pad=20)
plt.xlabel('Action Type', fontsize=12)
plt.ylabel('Queue 1 Values', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 4. Box Plots - Queue2
plt.figure(figsize=(12, 6))
bp2 = plt.boxplot([data[data['action'] == action]['queue2'] for action in sorted(colors.keys())],
                  labels=sorted(colors.keys()),
                  patch_artist=True)
for patch, color in zip(bp2['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
plt.title('Queue2 Distribution by Action', fontsize=14, pad=20)
plt.xlabel('Action Type', fontsize=12)
plt.ylabel('Queue 2 Values', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 5. Individual Scatter Plots
for action in colors:
    plt.figure(figsize=(10, 8))
    action_data = data[data['action'] == action]
    plt.scatter(action_data['queue1'],
               action_data['queue2'],
               c=colors[action],
               alpha=0.6,
               s=100)
    
    correlation = np.corrcoef(action_data['queue1'], action_data['queue2'])[0,1]
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8, pad=3),
             verticalalignment='top')
    
    plt.title(f'Queue Relationship - Action {action}', fontsize=14, pad=20)
    plt.xlabel('Queue 1', fontsize=12)
    plt.ylabel('Queue 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, 50)
    plt.ylim(0, 50)
    plt.tight_layout()
    plt.show()

# Print statistical summary
print("\nBasic Statistics:")
print("\nAction Counts:")
print(data['action'].value_counts().sort_index())
print("\nQueue1 Statistics by Action:")
print(data.groupby('action')['queue1'].describe())
print("\nQueue2 Statistics by Action:")
print(data.groupby('action')['queue2'].describe())
print("\nCorrelations by Action:")
for action in sorted(colors.keys()):
    action_data = data[data['action'] == action]
    corr = np.corrcoef(action_data['queue1'], action_data['queue2'])[0,1]
    print(f"Action {action}: {corr:.3f}")