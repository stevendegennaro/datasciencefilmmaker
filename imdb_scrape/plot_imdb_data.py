import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy.polynomial.polynomial as poly

def import_movies():
	print("Importing movies...")
	with open('data/imdbinfoformatted.json', 'r') as f:
		movie_list = json.load(f)

	print("Calculating star ratings...")
	imdb_ids = [d['imdb_id'] for d in movie_list]
	titles = [d['title'] for d in movie_list]
	imdb_stars = [d['imdb_stars'] for d in movie_list]
	lb_stars = [d['lb_stars'] for d in movie_list]

	df = pd.DataFrame(zip(titles,imdb_stars,lb_stars),
					  index = imdb_ids,
					  columns = ['title','imdb_stars','lb_stars'])

	return df

def plot_star_vs_star(df):

	fig, ax = plt.subplots()
	ax.scatter(df['imdb_stars'],df['lb_stars'],marker=".")
	ax.plot([0, 10], [0, 5], color='k', linestyle='-', linewidth=2)
	ax.scatter(df.loc['tt3449006','imdb_stars'],df.loc['tt3449006','lb_stars'],marker='*',color="red")
	ax.set_title("Letterboxd Stars vs IMDb User Rating")
	ax.set_xlabel("IMDb User Rating")
	ax.set_ylabel("Letterboxd Star Rating")
	plt.show()


def plot_star_polynomial(df, degree = 1):

	df_cleaned = df.dropna()

	c = poly.polyfit(df_cleaned['imdb_stars'], df_cleaned['lb_stars'], degree)
	print(c)
	x = np.arange(0,10,0.01)
	y = poly.polyval(x,c)

	fig, ax = plt.subplots()
	ax.scatter(df['imdb_stars'],df['lb_stars'],marker=".")
	ax.plot(x, y, color='k', linestyle='-', linewidth=2)
	ax.scatter(df.loc['tt3449006','imdb_stars'],df.loc['tt3449006','lb_stars'],marker='*',color="red")
	ax.set_title("Letterboxd Stars vs IMDb User Rating")
	ax.set_xlabel("IMDb User Rating")
	ax.set_ylabel("Letterboxd Star Rating")
	plt.show()

def plot_star_chi_by_eye(df):
	df_cleaned = df.dropna()
	c = poly.polyfit([1.13,8.55], [1.319,4.051], 1)
	print(c)
	x = np.arange(0,10,0.01)
	y = poly.polyval(x,c)

	fig, ax = plt.subplots()
	ax.scatter(df['imdb_stars'],df['lb_stars'],marker=".")
	ax.plot(x, y, color='k', linestyle='-', linewidth=2)
	ax.scatter(df.loc['tt3449006','imdb_stars'],df.loc['tt3449006','lb_stars'],marker='*',color="red")
	ax.set_title("Letterboxd Stars vs IMDb User Rating")
	ax.set_xlabel("IMDb User Rating")
	ax.set_ylabel("Letterboxd Star Rating")
	plt.show()

def plot_star_binned(df, n_bins = 25, degree = 1):
	df_cleaned = df.copy()
	df_cleaned.dropna(inplace=True)
	df_cleaned.drop('title',axis=1,inplace=True)

	bins = np.linspace(2, 8, n_bins)  # 50 bins between 0 and 5
	df_cleaned['bin'] = pd.cut(df_cleaned['imdb_stars'], bins)
	binned = df_cleaned.groupby('bin').mean()
	binned.dropna(inplace=True)
	# print(binned)

	c = poly.polyfit(binned['imdb_stars'], binned['lb_stars'], degree)
	print(c)
	x = np.arange(0,10,0.01)
	y = poly.polyval(x,c)

	fig, ax = plt.subplots()
	ax.scatter(df_cleaned['imdb_stars'],df_cleaned['lb_stars'],marker=".")
	# ax.scatter(binned['imdb_stars'],binned['lb_stars'],marker="o",color="black")
	ax.plot(x, y, color='k', linestyle='-', linewidth=2)
	ax.scatter(df.loc['tt3449006','imdb_stars'],df.loc['tt3449006','lb_stars'],marker='*',color="red")
	ax.set_title("Letterboxd Stars vs IMDb User Rating")
	ax.set_xlabel("IMDb User Rating")
	ax.set_ylabel("Letterboxd Star Rating")
	plt.show()

def plot_star_binned_weighted(df, n_bins = 25, degree = 1):
	df_cleaned = df.copy()
	df_cleaned.dropna(inplace=True)
	df_cleaned.drop('title',axis=1,inplace=True)

	bins = np.linspace(2, 8, n_bins)  # 50 bins between 0 and 5
	df_cleaned['bin'] = pd.cut(df_cleaned['imdb_stars'], bins)
	binned = df_cleaned.groupby('bin').agg(imdb_stars=('imdb_stars', 'mean'), lb_stars=('lb_stars', 'mean'), std_y=('lb_stars', 'std'))
	binned.dropna(inplace=True)
	# weights = 1 / binned['std_y']

	c = poly.polyfit(binned['imdb_stars'], binned['lb_stars'], w = 1/binned['std_y'], deg=degree)
	print(c)
	x = np.arange(0,10,0.01)
	y = poly.polyval(x,c)

	fig, ax = plt.subplots()
	ax.scatter(df_cleaned['imdb_stars'],df_cleaned['lb_stars'],marker=".")
	# ax.scatter(binned['imdb_stars'],binned['lb_stars'],marker="o",color="black")
	ax.plot(x, y, color='k', linestyle='-', linewidth=2)
	ax.scatter(df.loc['tt3449006','imdb_stars'],df.loc['tt3449006','lb_stars'],marker='*',color="red")
	ax.set_title("Letterboxd Stars vs IMDb User Rating")
	ax.set_xlabel("IMDb User Rating")
	ax.set_ylabel("Letterboxd Star Rating")
	plt.show()

def plot_star_linear_fixed_intercept(df):

	df_cleaned = df.copy()
	df_cleaned.dropna(inplace=True)
	# print(np.array(df_cleaned['imdb_stars']).reshape(-1,1))
	m, _, _, _ = np.linalg.lstsq(np.array(df_cleaned['imdb_stars']).reshape(-1,1), df_cleaned['lb_stars'], rcond=None)
	# print(m)
	x = np.arange(0,10,0.01)
	y = m * x

	fig, ax = plt.subplots()
	ax.scatter(df['imdb_stars'],df['lb_stars'],marker=".")
	ax.plot(x, y, color='k', linestyle='-', linewidth=2)
	ax.scatter(df.loc['tt3449006','imdb_stars'],df.loc['tt3449006','lb_stars'],marker='*',color="red")
	ax.set_title("Letterboxd Stars vs IMDb User Rating")
	ax.set_xlabel("IMDb User Rating")
	ax.set_ylabel("Letterboxd Star Rating")
	plt.show()

