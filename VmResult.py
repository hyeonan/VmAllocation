import matplotlib.pyplot as plt
import numpy as np

num_episodes = 100  # Update this with your number of episodes
average_allocation_percentages = []

for episode_number in range(num_episodes):
    data = np.loadtxt(f"OccupancyRate_Episode{episode_number+1}.txt", delimiter=",")
    # Make sure data is treated as scalar
    data = np.squeeze(data)
    data *= 100  # Convert data to percentage
    average_allocation_percentages.append(data)

plt.plot(range(num_episodes), average_allocation_percentages)
plt.axhline(y=50, color='r', linestyle='--')  # Add a horizontal line at 50%
plt.xlabel('Episode Number')
plt.ylabel('Average Allocation Percentage (%)')
plt.title('Average Allocation Percentage Across Episodes')
plt.show()

