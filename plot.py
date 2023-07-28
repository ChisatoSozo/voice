import matplotlib.pyplot as plt

# Read male scores
with open("male_scores.txt", "r") as f:
    male_scores = f.read().split(",")
    # Convert the scores to integer
    male_scores = [float(score)/10 for score in male_scores]

# Read female scores
with open("female_scores.txt", "r") as f:
    female_scores = f.read().split(",")
    # Convert the scores to integer
    female_scores = [float(score)/10 for score in female_scores]

# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Plot histogram for male scores
ax.hist(male_scores, color='blue', alpha=0.5, label='Male Scores', bins=20)

# Plot histogram for female scores
ax.hist(female_scores, color='pink', alpha=0.5, label='Female Scores', bins=20)

# Setting the legend
plt.legend(loc='upper right')

# Set a title
ax.set_title('Distribution of Scores by Gender (Female rating)')

# Set x-label
ax.set_xlabel('Scores')

# Set y-label
ax.set_ylabel('Frequency')

# Display the plot
plt.show()
