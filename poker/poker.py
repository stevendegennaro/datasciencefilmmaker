import pandas as pd
import statsmodels.api as sm
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats


def import_history() -> tuple[pd.DataFrame,pd.DataFrame]:
	filename = "poker_history.csv"
	poker_history = pd.read_csv(filename,header=0,index_col=0)
	counts_df = calculate_prizes(poker_history)
	return poker_history, counts_df

def calculate_prizes(poker_history):
	n_games = len(poker_history)
	bins = list(range(1,11))+[13,16,136]
	prizes = [153.90, 97.20, 70.20, 55.35, 40.50, 29.70, 20.25, 16.20, 12.15, 8.10, 6.75, 0]
	expected = ([1]* 9 + [3,3,135-15])
	expected = [x*n_games/sum(expected) for x in expected]

	poker_history['Prize'] = pd.cut(poker_history['Position'], bins=bins, labels=prizes, right = False)
	counts = poker_history['Prize'].value_counts().sort_index()
	counts_df = counts.reset_index(name='Actual')
	counts_df['Prize']= counts_df['Prize'].astype(float)
	counts_df['Expected'] = expected
	counts_df['Winnings'] = counts_df['Actual'] * counts_df['Prize']
	return counts_df

def calculate_winnings(positions):
	bins = list(range(1,11))+[13,16,136]
	prizes = np.array([153.90, 97.20, 70.20, 55.35, 40.50, 29.70, 20.25, 16.20, 12.15, 8.10, 6.75, 0])
	return prizes[np.digitize(positions,bins,right=False) - 1].sum() - 4.4*len(positions)


def plot_finishing_positions(poker_history):
	fig,ax = plt.subplots()
	poker_history.hist(ax = ax,column = "Position",bins = 135, edgecolor='black', linewidth=1)
	ax.set_ylabel("Frequency")
	ax.set_xlabel("Position")
	ax.set_title("Histogram of Finishing Positions")
	ax.set_xlim(0,135)
	ax.grid(axis='x', visible=False)
	ax.set_axisbelow(True)
	plt.show()


def plot_actual_vs_expected():
	poker_history, counts_df = import_history()
	ax = counts_df.plot(kind='bar')
	ax.set_title("Actual vs Expected Finishes for Each Prize Category")
	ax.set_ylabel("Count")
	ax.set_xticklabels([f"${label}" for label in counts_df.index],rotation=0)
	# plt.show()

def sim_run(n_games = 294):
	positions = np.random.randint(1,136,n_games)
	return calculate_winnings(positions)

def sim_many_runs(n = 1000):
	total_winnings = np.zeros(n)
	for i in range(n):
		total_winnings[i] = sim_run()
	return total_winnings

def qq_plot(total_winnings):
	fig = sm.qqplot(total_winnings, line='q')
	ax = plt.gca()
	ax.set_title(f"Q-Q Plot for {len(total_winnings):,} Runs")


def plot_runs(total_winnings, sums, nbins = 100):
	fig,ax = plt.subplots()
	plt.hist(total_winnings,bins = nbins,density=True)
	mu = total_winnings.mean()
	sigma = total_winnings.std()

	x = np.linspace(min(total_winnings), max(total_winnings), 1000)
	pdf = stats.norm.pdf(x, mu, sigma)
	plt.plot(x, pdf, 'black', linestyle = "--",linewidth=2)

	plt.hist(sums,bins = 100,density = True)
	plt.axvline(1379.40,color="blue")



	ax.set_title(f"Probability Distribution of Total Winnings\nAfter 294 Games ({len(total_winnings):,} Runs)")
	ax.set_xlabel("Total Winnings After 294 Games")
	ax.set_ylabel("Probability")


def calc_stats(total_winnings):
	mu = total_winnings.mean()
	sigma = total_winnings.std()
	count = len(total_winnings)
	skewness = stats.skew(total_winnings)

	print(f"Mean = {mu}")
	print(f"Standard Deviation = {sigma}")
	print(f"Loss per game = {mu/294}")
	print(f"Skew = {skewness}")


def resample_history(poker_history: pd.DataFrame, n_games = 294, n_samples = 100000):
	winnings = np.array(poker_history['Winnings'])
	resamples = np.random.choice(winnings, size=(n_samples, n_games), replace=True)
	sums = np.sum(resamples, axis=1)
	return sums



