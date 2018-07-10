# Import matplotlib.pyplot
import matplotlib.pyplot as plt
import pandas as pd

# Create and display the first scatter plot
hw = pd.read_csv('hourly_wages.csv')
hw.plot(kind='scatter', x='education_yrs', y='wage_per_hour')
plt.xlabel('education(in years)')
plt.ylabel('wages(per hour)')
plt.title('Hourly wages Vs Education')
plt.show()



# Add first subplot
plt.subplot(2, 1, 1)
gapminder = pd.read_csv('gapminder_tidy.csv')
# Create a histogram of life_expectancy
gapminder.life.plot(kind='hist')
# Group gapminder: gapminder_agg
gapminder_agg = gapminder.groupby('Year')['life'].mean()
plt.xlabel('Age(in years)')
plt.title('Histogram: Life Expentency')

# Add second subplot
plt.subplot(2, 1, 2)
# Create a line plot of life expectancy per year
gapminder_agg.plot()

# Add title and specify axis labels
plt.title('Life expectancy over the years')
plt.ylabel('Life expectancy')
plt.xlabel('Year')

# Display the plots
plt.tight_layout()
plt.show()

# Save both DataFrames to csv files
gapminder_agg.to_csv('gapminder_agg.csv',header=True)


