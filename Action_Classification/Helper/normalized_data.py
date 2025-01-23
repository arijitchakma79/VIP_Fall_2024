import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read the data
data = pd.read_csv('../Files/filtered_output.csv')

# Create new features
data['queue_ratio'] = data['queue1'] / (data['queue2'] + 1)  # Add 1 to avoid division by zero
data['queue_diff'] = data['queue1'] - data['queue2']
data['queue_product'] = data['queue1'] * data['queue2']
data['queue_sum'] = data['queue1'] + data['queue2']
data['queue1_squared'] = data['queue1'] ** 2
data['queue2_squared'] = data['queue2'] ** 2

# Set up colors
colors = {
    1: 'orange',
    2: 'blue',
    3: 'green',
    4: 'red',
    5: 'purple',
    6: 'brown',
    7: 'pink'
}

# 1. Original Features: Queue1 vs Queue2
plt.figure(figsize=(12, 8))
for action in colors:
    mask = data['action'] == action
    plt.scatter(data[mask]['queue1'], 
               data[mask]['queue2'],
               label=f'Action {action}',
               c=colors[action],
               alpha=0.6)
plt.title('Original: Queue1 vs Queue2', fontsize=14, pad=20)
plt.xlabel('Queue 1', fontsize=12)
plt.ylabel('Queue 2', fontsize=12)
plt.legend(title='Action Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 2. Queue Ratio vs Queue Difference
plt.figure(figsize=(12, 8))
for action in colors:
    mask = data['action'] == action
    plt.scatter(data[mask]['queue_ratio'], 
               data[mask]['queue_diff'],
               label=f'Action {action}',
               c=colors[action],
               alpha=0.6)
plt.title('Queue Ratio vs Queue Difference', fontsize=14, pad=20)
plt.xlabel('Queue Ratio (Queue1/Queue2)', fontsize=12)
plt.ylabel('Queue Difference (Queue1-Queue2)', fontsize=12)
plt.legend(title='Action Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 3. Queue Product vs Queue Sum
plt.figure(figsize=(12, 8))
for action in colors:
    mask = data['action'] == action
    plt.scatter(data[mask]['queue_product'], 
               data[mask]['queue_sum'],
               label=f'Action {action}',
               c=colors[action],
               alpha=0.6)
plt.title('Queue Product vs Queue Sum', fontsize=14, pad=20)
plt.xlabel('Queue Product (Queue1*Queue2)', fontsize=12)
plt.ylabel('Queue Sum (Queue1+Queue2)', fontsize=12)
plt.legend(title='Action Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 4. Squared Features
plt.figure(figsize=(12, 8))
for action in colors:
    mask = data['action'] == action
    plt.scatter(data[mask]['queue1_squared'], 
               data[mask]['queue2_squared'],
               label=f'Action {action}',
               c=colors[action],
               alpha=0.6)
plt.title('Queue1² vs Queue2²', fontsize=14, pad=20)
plt.xlabel('Queue1 Squared', fontsize=12)
plt.ylabel('Queue2 Squared', fontsize=12)
plt.legend(title='Action Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Print statistics for each feature combination
print("\nFeature Statistics by Action:")
for action in sorted(colors.keys()):
    action_data = data[data['action'] == action]
    print(f"\nAction {action}:")
    print(f"Count: {len(action_data)}")
    print(f"Ratio - Mean: {action_data['queue_ratio'].mean():.2f}, Std: {action_data['queue_ratio'].std():.2f}")
    print(f"Difference - Mean: {action_data['queue_diff'].mean():.2f}, Std: {action_data['queue_diff'].std():.2f}")
    print(f"Product - Mean: {action_data['queue_product'].mean():.2f}, Std: {action_data['queue_product'].std():.2f}")