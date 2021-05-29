# -*- coding: utf-8 -*-
"""
Created on Sun May 2 12:31:22 2021

@author: ramil.guluzada
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

results = pd.read_csv('C:/Users/RAMIL/Desktop/GNNCF/results/Results_agg1.csv')
results1 = pd.read_csv('C:/Users/RAMIL/Desktop/GNNCF/results/Results_agg2.csv')
results2 = pd.read_csv('C:/Users/RAMIL/Desktop/GNNCF/results/Results_agg3.csv')


def highlight_greaterthan(s, threshold, column):
    is_max = pd.Series(data=False, index=s.index)
    is_max[column] = s.loc[column] >= threshold
    return ['background-color: lightgreen' if is_max.any() else '' for v in is_max]

def plot_loss(df):
	sns.set(color_codes=True)
	sns.set_context("notebook", font_scale=1.)
	plt.figure(figsize=(15, 10))
	plt.subplot(2,2,1)
	plt.subplots_adjust(hspace=0.4)
	fig = sns.lineplot(x='loss', y='precision', hue='k', style='k',
		markers=True, markersize=10, linewidth=2, data=df.round(4), legend="full")
	fig.set(ylabel="PRECISION")
	fig.set(xlabel="BPR Loss")
	plt.subplot(2,2,2)
	fig = sns.lineplot(x='loss', y='recall', hue='k', style='k',
		markers=True, markersize=10, linewidth=2, data=df.round(4), legend="full")
	fig.set(ylabel="RECALL")
	fig.set(xlabel="BPR Loss")
	plt.subplot(2,2,3)
	fig = sns.lineplot(x='loss', y='hit_ratio', hue='k', style='k',
		markers=True, markersize=10, linewidth=2, data=df.round(4), legend="full")
	fig.set(ylabel="HIT_RATIO")
	fig.set(xlabel="BPR Loss")
	plt.subplot(2,2,4)
	fig = sns.lineplot(x='loss', y='ndcg', hue='k', style='k',
		markers=True, markersize=10, linewidth=2, data=df.round(4), legend="full")
	fig.set(ylabel="NDCG")
	fig.set(xlabel="BPR Loss")

def plot_loss1(df,df1):
	sns.set(color_codes=True)
	sns.set_context("notebook", font_scale=1.)
	plt.figure(figsize=(15, 10))
	plt.subplot(2,3,1)
	plt.subplots_adjust(hspace=0.4)
	fig = sns.boxplot(x='batch size', y='loss',data = df)
	fig.set(ylabel="BPR Loss")
	fig.set(xlabel="Batch size")
	plt.subplot(2,3,2)
	fig = sns.boxplot(x='batch size', y='training time',data = df)
	fig.set(ylabel="Training time")
	fig.set(xlabel="Batch Size")
	plt.subplot(2,3,3)
	fig = sns.boxplot(x='batch size', y='hit_ratio', data=df)
	fig.set(ylabel="HIT_RATIO")
	fig.set(xlabel="Batch size")


	plt.subplot(2,3,4)
	plt.subplots_adjust(hspace=0.4)
	fig = sns.boxplot(x='lr', y='Loss',data = df1)
	fig.set(ylabel="BPR Loss")
	fig.set(xlabel="learning rate")
	plt.subplot(2,3,5)
	fig = sns.boxplot(x='lr', y='NDCG',data = df1)
	fig.set(ylabel="NDCG")
	fig.set(xlabel="learning rate")
	plt.subplot(2,3,6)
	fig = sns.boxplot(x='lr', y='Recall', data=df1)
	fig.set(ylabel="Recall")
	fig.set(xlabel="learning rate")

plot_loss(results)

plot_loss1(results1,results2)
plt.show()