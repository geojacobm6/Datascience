import pandas as pd
import numpy as np
import matplotlib as plt


df = pd.read_csv("train.csv") #Reading the dataset in a dataframe using Pandas

print(df.head(10))
print(df.describe())
print(df['Property_Area'].value_counts())
print(df['ApplicantIncome'].hist(bins=50))
print(df.boxplot(column='ApplicantIncome'))


temp1 = df['Credit_History'].value_counts(ascending=True)
temp2 = df.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
print 'Frequency Table for Credit History:'
print temp1

print '\nProbility of getting loan for each Credit History class:'
print temp2


fig = plt.figure.Figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit_History")
temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by credit history")

temp3 = pd.crosstab(df['Credit_History'], df['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)