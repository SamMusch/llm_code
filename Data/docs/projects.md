---
title: projects
created: '2025-10-12'
modified: '2025-10-12'
source_file: projects.md
word_count: 410
reading_time: 2.0
children: 0
grandchildren: 0
ai_abstract: null
ai_key_terms: []
_kMDItemDisplayNameWithExtensions: projects.md
kMDItemContentCreationDate: 2025-10-07 22:23:24 +0000
kMDItemContentCreationDate_Ranking: 2025-10-07 00:00:00 +0000
kMDItemContentModificationDate: 2025-10-12 14:43:31 +0000
kMDItemContentType: net.daringfireball.markdown
kMDItemContentTypeTree: (
kMDItemDateAdded: 2025-10-07 22:23:24 +0000
kMDItemDocumentIdentifier: '222829'
kMDItemFSCreatorCode: ''
kMDItemFSFinderFlags: '0'
kMDItemFSHasCustomIcon: (null)
kMDItemFSInvisible: '0'
kMDItemFSIsExtensionHidden: '0'
kMDItemFSIsStationery: (null)
kMDItemFSLabel: '0'
kMDItemFSNodeCount: (null)
kMDItemFSOwnerGroupID: '20'
kMDItemFSOwnerUserID: '502'
kMDItemFSTypeCode: ''
kMDItemInterestingDate_Ranking: 2025-10-12 00:00:00 +0000
Due: null
Function: null
Objective: null
Quality: null
QualityComment: null
ReviewFreq: null
CoverImage: null
HoursDone: null
HoursRemain: null
tags: null
TimeSpent: null
TimeSpent2: null
Covers: null
cssclasses: null
aliases: null
---

- Time Series Multivariate Deep Learning with LSTM
    - [Code in Google Colab](https://colab.research.google.com/drive/14Z3BsEq12YfcDTmO_OJi9YmiLhpm2bw_#scrollTo=29e2r-xyXrQN)
    - [[ML-TS]] - Notes

This was a project I completed after taking a deep-dive into LSTMs. The objective is to predict daily household electricity consumption given 4 years of data. 

---

- Time Series Multivariate Customer Forecasting (LSTM, LightGBM)
    - [Report with Code](https://htmlpreview.github.io/?https://github.com/SamMusch/Predictive-Project-Time-Series/blob/master/Predictive%20KT.html)
    - [Presentation Slides](https://docs.google.com/presentation/d/1bUKSU8vLlv2M4-dflHaJGDqiRqiWalr-/edit?usp=sharing&ouid=111023174892277357363&rtpof=true&sd=true)
    - [Github Repository](https://github.com/SamMusch/Predictive-Project-Time-Series)

This was a project we completed for our course in Predictive Analytics (supervised learning) in Fall 2019. We were looking to forecast the number of daily visitors for 150 different restaurants located in different parts of Japan. Our final model was an ensemble of RNN (sequential neural network model) and LightGBM (faster than XGboost to handle our large dataset).

---

- Marketing (A/B Testing, Multivariate Causal Inference)
    - [Final Report with Code](https://htmlpreview.github.io/?https://github.com/SamMusch/00-Data-Science/blob/main/Marketing%20Causal%20Analysis.html)
    - [Kaggle Dataset](https://www.kaggle.com/bletchley/bank-marketing#balanced_bank.csv)

This repository includes the project our team performed for Causal Inference. Our dataset compared 2 marketing campaigns against a control group. We used a technique called “matching” to make sure that we were comparing people against those who were similar to them, and we also used a technique called “difference in difference” to evaluate how the campaigns differed after a surprise news story.

---

- Minneapolis Crime (Sentiment Analysis, XGBoost, AWS)
    - [Github](https://github.com/SamMusch/Minneapolis-Crime)
    - [Tableau](https://public.tableau.com/profile/sam.musch#!/vizhome/MinneapolisCrime/Sheet1)

This was the project our team completed for our course in Big Data in Fall 2019. We were looking to predict the number of crimes that would occur for each of the 5 police precincts on a daily level to help improve the number of people staffed on the day.

We used AWS services such as S3 for storage, Comprehend for sentiment analysis, and SageMaker to run our XGboost predictive model. We also used Tableau to dig deeper into the details of what had happened in the past and display geographic information. Our final model resulted in [25% average error](https://i.imgur.com/8ow32Gy.pnghttps://i.imgur.com/8ow32Gy.png) (MAPE), cutting the current error in half.

