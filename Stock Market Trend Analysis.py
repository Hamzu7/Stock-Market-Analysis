#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


data = pd.read_excel("Stock.xlsx")
data


# In[5]:


data.isnull().sum()


# In[6]:


data.info()


# In[7]:


data.rename(columns={"Date":"Datetime"}, inplace=True)
data


# In[8]:


plt.figure(figsize=(14,6))
plt.plot(data["Datetime"], data["Apple (AAPL)"], label="Apple (AAPL)", color='red')
plt.plot(data["Datetime"], data["Alphabet (GOOGL)"], label="Alphabet (GOOGL)", color='green')
plt.plot(data["Datetime"], data["Bank of America (BAC)"], label="Bank of America (BAC)", color='blue')

plt.title("Stock price over time")
plt.xlabel("Date")
plt.ylabel("Price USD")
plt.legend()
plt.grid(True)
plt.tight_layout
plt.show()


# In[9]:


data['AAPL_7day_avg'] = data['Apple (AAPL)'].rolling(window=7).mean()
data['GOOGL_7day_avg'] = data['Alphabet (GOOGL)'].rolling(window=7).mean()
data['BAC_7day_avg'] = data['Bank of America (BAC)'].rolling(window=7).mean()
data


# In[10]:


data["AAPL_Return"] = data["Apple (AAPL)"].pct_change() * 100
data["GOOGL_Return"] = data["Alphabet (GOOGL)"].pct_change() * 100
data["BAC_Return"] = data["Bank of America (BAC)"].pct_change() * 100
data


# In[11]:


prices = data[['Apple (AAPL)', 'Alphabet (GOOGL)', 'Bank of America (BAC)']]

corr_matrix = prices.corr()

plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Between Stocks')
plt.tight_layout()
plt.show()


# In[12]:


print("Volatility (std dev of daily returns):")
print("Apple:", data['AAPL_Return'].std())
print("Google:", data['GOOGL_Return'].std())
print("Bank of America:", data['BAC_Return'].std())


# # Insights & Observations

# ### Top Performing Stock:
# Alphabet (GOOGL) consistently showed the highest price growth, making it the top performing stock during the given time period.

# ### Highest Daily Change:
# Bank of America (BAC) showed the highest daily fluctuations, reflecting higher volatility in its price movements.

# ### Lowest Daily Change:
# Alphabet (GOOGL) had the lowest volatility, indicating more stable price trends.

# ### Volatility Comparison (Std. Dev of Daily Returns):
# 
# Apple (AAPL): ~0.97
# 
# Google (GOOGL): ~0.83
# 
# Bank of America (BAC): ~1.51 ← most volatile

# ### Correlation Insights (from Heatmap):
# 
# Apple & Google: Strong positive correlation (0.79) → similar market behavior
# 
# Bank of America had low correlation with both tech stocks → behaves independently

# ### Investment Advice (Data based only):
# 
# For stability: Alphabet (GOOGL) is best due to low volatility and consistent growth.
# 
# For high-risk/high-return: Bank of America (BAC) could be considered, but it's riskier due to volatility.
# 
# Apple (AAPL) lies somewhere in between both.

# # Conclusion:
#     
# This analysis helped me understand stock trends using Python (pandas, matplotlib, seaborn).
# 
# Data shows tech stocks (Apple & Google) are more correlated, while financials (BAC) behave differently.
# 
# Rolling averages, daily return percentages, and volatility gave clear insights into performance and risk.

# ## Next Steps / Future Work:
#     
# Include more stocks and sectors for broader analysis
# 
# Perform monthly or quarterly trend breakdowns
# 
# Predict future prices using ML models
