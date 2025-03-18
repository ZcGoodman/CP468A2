import matplotlib.pyplot as plt
import csv
import math
import random 

random.seed(42)

def most_frequent(List):
    return max(set(List), key=List.count)

# Define the knn function to compute accuracy
def knn(data, new, k, percent):
    correct_predictions = 0
    total_test_points = len(new[0])
    
    for i in range(len(data[0])):  # Adjusted for proper index access
        plt.scatter(data[0][i], data[1][i], c=data[2][i], s=21)
    
    while len(new[0]) != 0: 
        x1 = new[0][0]
        y1 = new[1][0]
        true_label = new[2][0]  # True label of the test point
        distance = []
        
        for j in range(len(data[0])):
            x2 = data[0][j]
            y2 = data[1][j]
            sum = math.sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2))
            distance.append(float(sum))
        
        # Get k-nearest neighbors
        min_indices = sorted(enumerate(distance), key=lambda x: x[1])[:k]
        min_positions = [x[0] for x in min_indices]  # Their indices

        color_index = [data[2][z] for z in min_positions]
        predicted_label = most_frequent(color_index)    

        if predicted_label == true_label:
            correct_predictions += 1  # Count correct predictions

        plt.scatter(x1, y1, c=predicted_label, edgecolors='black')
        
        # Update training set
        new[0].pop(0)
        new[1].pop(0)
        new[2].pop(0)
        data[0].append(float(x1))
        data[1].append(float(y1))
        data[2].append(predicted_label)

    plt.title(f"Customer Behavior: k-NN (k={k}) (split={percent}%)")
    plt.xlabel("Average amount spent per Purchase (in $)")
    plt.ylabel("Frequency of purchases per month")
    plt.legend(["Not interested", "Interested"], loc="lower right")
    plt.ylim(0, 3)
    plt.show()
    
    accuracy = correct_predictions / total_test_points
    return accuracy


# Read the CSV file and process the data
filename = "CustomerDataset_Q1.csv"
rows = []

with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)  # Skip headers
    
    for row in csvreader:
        rows.append(row)

# Process dataset
x1, x2, y = [], [], []

for row in rows:
    x1.append(float(row[0]))  # X coordinate
    x2.append(float(row[1]))  # Y coordinate
    y.append('blue' if row[2] == '0' else 'red')  # Convert to colors

# Define different splits
splits = [(16, 4, 80), (12, 8, 60), (10, 10, 50)]
k_values = [1, 2, 3, 4]

# Store results
results = {}

for k in k_values:
    results[k] = {}
    for train_size, test_size, percent in splits:
        # Create training and test datasets
        x1_train, x2_train, y_train = x1[:train_size], x2[:train_size], y[:train_size]
        x1_test, x2_test, y_test = x1[-test_size:], x2[-test_size:], y[-test_size:]

        dataset_train = [x1_train, x2_train, y_train]
        dataset_test = [x1_test, x2_test, y_test]

        accuracy = knn(dataset_train, dataset_test, k, percent)
        results[k][percent] = accuracy

# Print accuracy results
print("\nAccuracy Table:")
print(" k  |  80% Train  |  60% Train  |  50% Train")
print("------------------------------------------")
for k, accs in results.items():
    print(f" {k}  |  {accs[80]:.2f}       |  {accs[60]:.2f}       |  {accs[50]:.2f}")
