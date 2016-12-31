
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn import linear_model
import seaborn
from datetime import datetime
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import mstats
from scipy.stats import pearsonr
from scipy.stats import norm
#get_ipython().magic(u'matplotlib inline')
import statsmodels.api as sm
#import statsmodels.formula.api as smf
from statsmodels.graphics.api import abline_plot
import scipy.stats as stats
from scipy.stats import mstats
from sklearn import linear_model
from sklearn.metrics import r2_score
#from __future__ import division
import matplotlib.pyplot as plt
from minepy import MINE
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import os, errno
# Generate some data for this demonstration.
#data = norm.rvs(10.0, 2.5, size=500)


class DataSciency(object):

	def create_path_if_doesnt_exist(path_to_file):
	    if not os.path.exists(os.path.dirname(path_to_file)):
	        try:
	            os.makedirs(os.path.dirname(path_to_file))
	        except OSError as exc: # Guard against race condition
	            if exc.errno != errno.EEXIST:
	                raise

	def plot_OLS_CI(self, model, x, y, y_true):
		prstd, iv_l, iv_u = wls_prediction_std(model)
		fig, ax = plt.subplots(figsize=(8,6))

		ax.plot(x, y, 'o', label="data")
		ax.plot(x, y_true, 'b-', label="True")
		ax.plot(x, model.fittedvalues, 'r--.', label="OLS")
		ax.plot(x, iv_u, 'r--')
		ax.plot(x, iv_l, 'r--')
		ax.legend(loc='best');

	# testing normality
	def visualize_normality(self, mu, sigma, variable):
		#mu, sigma = 0, 0.1 # mean and standard deviation
		s = np.random.normal(mu, sigma, 1000)
		#Verify the mean and the variance:
		if not (abs(mu - np.mean(s)) < 0.01):
			print "normality verification 1 failed"
		if not abs(sigma - np.std(s, ddof=1)) < 0.01: #ddof=len(variable)-1)) < 0.01:
			print "normamlity verification 2 failed"
		#Display the histogram of the samples, along with the probability density function:
		count, bins, ignored = plt.hist(s, 30, normed=True)
		plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
			np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
		linewidth=2, color='r')
		plt.show()

	def visualize_normality_for_sample(self, x):
		#Display the histogram of the samples, along with the probability density function:
		mu =  x.mean()
		sigma = x.std()
		count, bins, ignored = plt.hist(x, 30, normed=True)
		plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
			np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
		linewidth=2, color='r')
		plt.show()

	def test_normality(self, data):
		"""
		Tests whether  a sample differs from a normal distribution. Returns a 2-tuple of the chi-squared statistic,
		and the associated p-value. Given the null hypothesis that x came from a normal distribution,
		If the p-val is very small (alpha level of 0.05 normally), it means it is unlikely that the data came from a normal distribution.
		Other possible way: https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.chisquare.html
		"""
		# equivalent: print stats.normaltest(data)
		print "z value and p value: "#, z, pval
		z,pval = mstats.normaltest(data)
		if(pval < 0.05):
			print "Not normal distribution"
		return z, pval

	# normalization of categorical features
	def create_dummy_vars_for_categorical_features(self, train_df, categorical_features):
		print "Before dummys:\n", train_df.shape ,' ',train_df.columns.values, '\n', train_df.head()
		new_df = train_df.copy()
		for f in categorical_features:
			num_uniques  = len(train_df[f].unique())
			df = pd.get_dummies(train_df[f],prefix = f).iloc[:,:num_uniques-1]
			train_df = pd.concat([train_df, df],axis=1)
		new_df.drop(categorical_features,axis=1,inplace=True)
		print "After dummys:\n", df.shape ,' ',train_df.columns.values, '\n',df.head()
		return new_df

	def find_correlated_features(self, df_full, y, correlation_coef, cols_to_skip=[]):
		# selecting features to perform pairwise correlation analysis
		print "initial columns ",df_full.columns.values
		# selecting features to perform pairwise correlation analysis
		cols_to_consider = list(df_full.columns.values.copy())
		for s in cols_to_skip:
			cols_to_consider.remove(s)
		for c in df_full.columns.values :
			if c.startswith('1_'):
				cols_to_consider.remove(c)
		print "cols_to_consider", cols_to_consider
		df =  df_full.loc[:, cols_to_consider]
		print "Df to find correlation analysis: \n", df.columns.values, '\n', df.head(5)

		col_names = df.columns.values
		corr_list = list()
		for col in range(df.values.shape[1]):
			x = df.iloc[:,col]
			_pear =  pearsonr(x, y) # returns the pearson r correlation value, and the p-value associated
			# also option b) x.corr(y, method='spearman')
			#print "Column name and Pearson correlation with Y (r and p-value): ",col_names[col], _pear
			corr_list.append(_pear[0])

		corr_list = np.array(corr_list)
		sorting_index = np.argsort(-corr_list)
		corr_list = corr_list[sorting_index]
		col_names = col_names[sorting_index]
		selected_features = list()
		for (cr, col) in  zip(corr_list,col_names):
			if abs(cr) > 0.05:
				selected_features.append((col, cr))
		print len(selected_features), " features with >0.05 corr coef with Y: \n"
		for f in selected_features:
			print f

		additional_features = list()
		mut_correlated_features = []
		highly_corr_features = []
		# only if not actually doing feature selection:
		selected_features = col_names
		for i in range(len(selected_features)):
			for j in range(len(selected_features)):
				if j > i and not (selected_features[i].startswith('1_') and selected_features[j].startswith('1_')):
					# adding linear combination of features to improve the model
					feat1_ = selected_features[i]
					feat2_ = selected_features[j]
					# comb_feat = feat1_ + '_' + feat2_
					# additional_features.append(comb_feat)
					# df[comb_feat] = df[feat1_]*df[feat2_]
					# detecting features that are correlated with each other
					# returns the pearson r correlation value, and the p-value associated
					_pear =  df[feat1_].corr(df[feat2_], method=correlation_coef) #pearsonr(train[feat1_], train[feat2_])
					if abs(_pear) > 0.01 :
						print "Columns HIGHLY correlated (r and p-value) for X1 and X2 ",feat1_, " and ", feat2_, ": ", _pear
						highly_corr_features.append((feat1_, feat2_, _pear))
					else:
						print "Columns correlation (r and p-value) for ",feat1_, " and ", feat2_, ": ", _pear
					# 	mut_correlated_features.append((feat1_, feat2_, _pear[0], _pear[1]))
	  #   for correlation_tuple in highly_corr_features:
			# print "Highly correlated variables corr coeff: ",correlation_tuple
		print "Highly correlated features among themselves: ",len(highly_corr_features)
		for f in highly_corr_features:
			print f

	def visualize_feature_correlations(self, df_full, correlation_coef, cols_to_skip=[]):
		"""
		correlation_coef: 'pearson, 'kendall', and 'spearman' are supported. Use pearson when normally distributed data and spearman
		and kendall when linearity between x and y is observed. Spearman is faster to compute but
		Kendall is more reliable and interpretable
		Correlations are returned in a new DataFrame instance (corr_df below).
		"""
		print "initial columns ",df_full.columns.values
		# selecting features to perform pairwise correlation analysis
		cols_to_consider = list(df_full.columns.values.copy())
		for s in cols_to_skip:
			cols_to_consider.remove(s)
		for c in df_full.columns.values :
			if c.startswith('1_'):
				cols_to_consider.remove(c)
		print "cols_to_consider", cols_to_consider
		df =  df_full.loc[:, cols_to_consider]
		print "Df to perform correlation analysis: \n", df.columns.values#, '\n', df.head(5)

		# These settings modify the way  pandas prints data stored in a DataFrame.
		# In particular when we use print(data_frame_reference); function - all
		#  column values of the frame will be printed in the same  row instead of
		# being automatically wrapped after 6 columns by default. This will be
		# for looking at our data at the end of the program.
		#pd.set_option('display.height', 1000)
		pd.set_option('display.max_rows', 500)
		pd.set_option('display.max_columns', 500)
		pd.set_option('display.width', 1000)

		corr_df = df.corr(method=correlation_coef)
		print "--------------- CORRELATIONS ---------------"
		print corr_df.head(len(df.columns))

		print "--------------- CREATE A HEATMAP ---------------"
		# Create a mask to display only the lower triangle of the matrix (since it's mirrored around its
		# top-left to bottom-right diagonal).
		mask = np.zeros_like(corr_df)
		mask[np.triu_indices_from(mask)] = True

		# Create the heatmap using seaborn library.
		# List if colormaps (parameter 'cmap') is available here: http://matplotlib.org/examples/color/colormaps_reference.html
		seaborn.heatmap(corr_df, cmap='RdYlGn_r', vmax=1.0, vmin=-1.0 , mask = mask, linewidths=2.5)

		# Show the plot we reorient the labels for each column and row to make them easier to read.
		plt.yticks(rotation=0)
		plt.xticks(rotation=90)
		plt.show()

	def get_xy_reshaped_for_numpy(self, data):
		# an array with dimensions (x,1) and (x,) are treated differently by numpy
		x=data.iloc[:,0]
		y=data.iloc[:,1]
		m,n =np.shape(data)
		x = x.values.reshape(m,1)
		y = y.values.reshape(m,1)
		#x=np.c_[np.ones((m,1)),X]
		#print "new shape: ", x.shape, y.shape
		return x, y

	def get_reshaped_for_numpy(self, x):
		# an array with dimensions (x,1) and (x,) are treated differently by numpy
		#data = np.atleast_2d(x)
		#print "new shape: ", x.shape
		return np.atleast_2d(x).T

	def summarize_stats(self, y):
		mean =  y.mean()
		std = y.std()
		median = np.nanmedian(y)
		print "Feature mean: ", mean
		print "Feature std dev: ", std
		print "Feature std error of the mean: ", stats.sem(y)
		print "Feature median: ", median

	def impute_column_data(self, data):
		#impute the data with any of the two lines below
		#if data.isnull().any():
			# a) bad_indices = np.where(np.isinf(test))
			# You can then use those indices to replace the content of the array:
			#test[bad_indices] = -1
			#b
		return data.fillna(lambda x: x.median())
			#print "impute_column_data with ffill"
		#return data.fillna(method='ffill')
		# else:
		# 	return data

	def update_score_and_params(self, results, filename, params, r2, model):
		if len(params)>1:
			results.append({'filename': filename, 'a': params[0], 'b': params[1], 'r2': r2, 'model': str(model)})
		else:
			results.append({'filename': filename, 'a': params[0], 'b':0, 'r2': r2, 'model': str(model)})
		return results

	# def update_score_and_params(self, max_score, params, name, model, current_score):
	#     if current_score > max_score:
	#         max_score = current_score
	#         max_score_params = params
	#         best_model = model
	#         best_name = name
	# 	return max_score, max_score_params, name, model

	def histogram_of_std_deviance_residuals(self, x, y, fit_model):
		"""
		### Plots
		# TODO:UNTESTED FOR ALL EXPONENTIAL FAMILY FUNCTIONS
		"""
		nobs = fit_model.nobs #		print "n observations: ",nobs
		y = y/y.sum(1) #data.endog[:,0]/data.endog.sum(1)
		yhat = fit_model.mu

		# Plot yhat vs y:
		from statsmodels.graphics.api import abline_plot

		fig, ax = plt.subplots()
		ax.scatter(yhat, y)
		line_fit = sm.OLS(y, sm.add_constant(yhat, prepend=True)).fit()
		abline_plot(model_results=line_fit, ax=ax)

		ax.set_title('Model Fit Plot')
		ax.set_ylabel('Observed values')
		ax.set_xlabel('Fitted values');

		#Plot yhat vs. Pearson residuals:
		fig, ax = plt.subplots()

		ax.scatter(yhat, fit_model.resid_pearson)
		ax.hlines(0, 0, 1)
		ax.set_xlim([0, 1])
		ax.set_title('Residual Dependence Plot')
		ax.set_ylabel('Pearson Residuals')
		ax.set_xlabel('Fitted values')

		#Histogram of standardized deviance residuals:
		from scipy import stats
		fig, ax = plt.subplots()
		resid = fit_model.resid_deviance.copy()
		resid_std = stats.zscore(resid)
		ax.hist(resid_std, bins=25)
		ax.set_title('Histogram of standardized deviance residuals');

		# QQ Plot of Deviance Residuals:

		from statsmodels import graphics
		fig = graphics.gofplots.qqplot(resid, line='r')

	def model_fit_plot(self, y, yhat):
		fig, ax = plt.subplots()
		ax.scatter(yhat, y)
		line_fit = sm.OLS(y, sm.add_constant(yhat, prepend=True)).fit()
		abline_plot(model_results=line_fit, ax=ax)
		ax.set_title('Model Fit Plot')
		ax.set_ylabel('Observed values')
		ax.set_xlabel('Fitted values')
		ax.set_title('Model fit plot')

	def fit_normal(self, y):
		# Fit a normal distribution to the data:
		mu, std = norm.fit(y)
		# Plot the histogram.
		plt.hist(y, bins=25, normed=True, alpha=0.6, color='g')

		# Plot the PDF.
		xmin, xmax = plt.xlim()
		x = np.linspace(xmin, xmax, 100)
		p = norm.pdf(x, mu, std)
		plt.plot(x, p, 'k', linewidth=2)
		title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
		plt.title(title)
		plt.show()

	def scatter_plot(self, x,y, title):
		fig, ax = plt.subplots()#figsize=(8,6))
		ax.legend(loc='best')
		ax.scatter(x,y, alpha=0.5, s=20)#(x, c=close, alpha=0.5) #s = ms= markersize
		ax.set_xlabel('X')#,fontsize=20)
		ax.set_ylabel('Y')#,fontsize=20)
		ax.set_title('Scatter plot')
		plt.scatter(x,y,s=20)
		plt.title(title)
		plt.show()

	def OLS_nonlinear_curve_but_linear_in_params(self, x, y):
		#Non-linear relationship between x and y
		# nsample = 50
		# sig = 0.5
		# x = np.linspace(0, 20, nsample)
		# X = np.c_[x, np.sin(x), (x - 5)**2, np.ones(nsample)]
		# beta = [0.5, 0.5, -0.02, 5.]
		# y_true = np.dot(X, beta)
		# y = y_true + sig * np.random.normal(size=nsample)
		fit_model = sm.OLS(y, y).fit()
		print fit_model.summary()

	# for computing correlation in between categorical features, use Mutual Information coefficient, e.g., using
	# package minepy (Maximal Information-based Nonparametric Exploration)
	def plot_covariance_based_mutual_info_for_categorical_correlations(self, df, title_id, categorical_features = []):
		"""
		Computes covariance matrix using a vectorized implementation to be used for computing the mutual information
		coefficient
		"""
		cols = []
		plot_id = 1
		for c in df.columns:
			if not c.startswith('1_') and c !='0':
				cols.append(c)
		plot_grid_wide = len(cols)#/2
		plot_grid_length = len(cols)#/plot_grid_wide
		print "Computing covariance matrix and MIC for features: ",cols
		for i in range(len(cols)):
			for j in range(len(cols)):
				if j > i and not (cols[i].startswith('1_') and cols[j].startswith('1_')):
					#cov_matrix = np.cov([df[cols[i]], df[cols[j]]], ddof= 0)
					self.MIC_plot(df[cols[i]], df[cols[j]], plot_grid_wide, plot_grid_length, plot_id, cols[i], cols[j], title_id)
					plot_id +=1

		plt.figure(facecolor='white')
		#plt.tight_layout()
		plt.show()

	def MIC_plot(self, x, y, numRows, numCols, plotNum, x_name, y_name, filename):
		# build the MIC and correlation plot using the covariant matrix using a vectorized implementation. To be used when
		# categorical features are part of the model (otherwise, Pearson, Kendall and Spearman can be used)
		print "Pearson product-moment correlation coefficients np.corrcoef(x=",x_name,", y=",y_name,"): ",np.corrcoef(x, y)
		r = np.around(np.corrcoef(x, y)[0, 1], 1)  # Pearson product-moment correlation coefficients.
		# TODO: compute cov matrix for each one-hot encoding variable of the categorical feature with
		# MINE's Mutual Information coefficient

		fig = plt.figure(figsize=(33,5), frameon=True)#, ms=50)
		mine = MINE(alpha=0.6, c=15, est="mic_approx")
		mine.compute_score(x, y)
		mic = np.around(mine.mic(), 1)
		ax = plt.subplot(numRows, numCols, plotNum)
		ax.set_xlim(xmin=min(x)+1, xmax=max(x)+1)
		ax.set_ylim(ymin=min(y)+1, ymax=max(y)+1)
		ax.set_title('Pearson r=%.1f\nMIC=%.1f Features %s and %s in %s' % (r, mic, x_name, y_name, filename),fontsize=10)
		ax.set_frame_on(False)
		ax.axes.get_xaxis().set_visible(True)
		ax.axes.get_yaxis().set_visible(True)
		ax.plot(x, y, '*')
		plt.xlabel('X')
		plt.ylabel('Y')
		# ax.set_xticks([])
		# ax.set_yticks([])
	#     plt.scatter(x,y,s=s)
	#     plt.show()
		return ax

	def fit_model(self, model, x, y, filename):
		fitted_model = model.fit(x,y)
		#print filename," coefs: ",model.coef_, " intercept: ",model.intercept_
		return fitted_model #model.coef_, model.intercept_

	def plot_value_count_histogram(self, y, title):
		counts = y.value_counts()
		counts.hist(bins=[0, 1, 2, 3,4, 5])
		print counts.head()
		fig = plt.figure()
		#ax = fig.add_subplot(111, projection='2d')
		plt.title('Value counts histogram '+title)
		plt.xlabel('Count')
		plt.ylabel('Values')
		plt.show()

	def plot_histogram(self, y, title):
		plt.hist(y) #bins=[0, 1, 2, 3,4,5]
		plt.title('Feature histogram')
		plt.xlabel('Y')
		plt.ylabel('Value count')
		plt.title(title)
		plt.show()

	def test_normality(self, data):
		"""
		Tests whether a sample differs from a normal distribution. Returns a 2-tuple of the chi-squared statistic,
		and the associated p-value. Given the null hypothesis that x came from a normal distribution,
		If the p-val is very small (alpha level of 0.05 normally), it means it is unlikely that the data came from a normal distribution.
		Other possible way: normal probab. plot and https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.chisquare.html
		"""
		# equivalent: print stats.normaltest(data)
		print "z value and p value: "#, z, pval
		z,pval = mstats.normaltest(data)
		if(pval < 0.05):
			print "Not normal distribution"
		else:
			print "It is a Normal Distribution"
		return z, pval

	def fit_fat_tails_model(self, x,y):
	    # Standardized Median Absolute Deviation (MAD) is a consistent estimator for sigma_hat
	    #sigma_hat=K * MAD where K depends on the distribution. For the normal distribution for example,
	    #\[K = \Phi^{-1}(.75)\]
	    stats.norm.ppf(.75)
	    MAD = sm.robust.scale.stand_mad(x)
	    #The default for Robust Linear Models is MAD. Another popular choice is Huber's proposal 2
	    np.random.seed(12345)     # example: fat_tails = stats.t(6).rvs(40)
	    # Univariate Kernel Density Estimator (KDE)
	    #kde = sm.nonparametric.KDE(fat_tails)
	    kde = sm.nonparametric.KDEUnivariate(y)
	    #kde.fit(kernel, bw, fft, gridsize=gridsize, cut=cut, clip=clip)
	    fitted = kde.fit()
	    fig = plt.figure()
	    ax = fig.add_subplot(111)
	    ax.plot(kde.support, kde.density)
	    ax.set_ylabel('Density')
	    ax.set_xlabel('Support')
	    ax.set_title('Fat tails model with Univariate kernel density estimator (KDE) and MAD')
	    return fitted, MAD

	def fit_fat_tails_to_test(self, x,y):
		""" Other options to explore in future:
    
	    #residuals = d.get_reshaped_for_numpy(model.resid)
	    print filename,' OLS Parameters: ', params, ' R2: ', r2#, 'len(residuals): ',residuals.shape
	    print(model.summary())
	    # not implemented y_pred = OLS.fit(y, x).predict(x)
	    #    print x.shape, y.shape
	    #    y_pred = OLS.predict(x)
	    #     y_pred = y_pred.values.reshape(m,1) # ValueError: shapes (100,1) and (100,) not aligned: 1 (dim 1) != 100 (dim 0)
	    #     r2_score = r2_score(y, y_pred)
	    #     print("r^2 on test data : %f" % r2_score)
	    #     residuals = y_pred - y
	    #   d.model_fit_plot(Y, y_pred)
	    #d.summarize_stats(residuals)
	    #print "residuals normality test: \n", d.test_normality(residuals)
	    #d.scatter_plot(x, residuals, "Residuals")
	    
	    update_best_model_score_and_params("OLS", model, params, r2, max_score)
	    filenames_log.append(filename); a_params.append(params[0]); b_params.append(params[1]); r2_logs.append(r2); model_logs.append(model);
	    #plot_OLS_CI(results, x, y, y_true)
	    #d.histogram_of_std_deviance_residuals(x, y, model)
	    """
	    print "toDO"


if __name__ == '__main__':	
	x, y = d.get_xy_reshaped_for_numpy(data)
	# Univariate Kernel Density Estimator (KDE) with Standardized Median Absolute Deviation (MAD)
	model, MAD = fit_fat_tails_model(x, y)
	#filenames_log.append(filename); params_log.append(params); r2_logs.append(r2); model_logs.append(model); epsilons.append(-1)

