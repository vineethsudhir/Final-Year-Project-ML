import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

Label_encoder = preprocessing.LabelEncoder()

#file names defined here
time_table_file = 'timetable.csv'
survey_file = 'surveysumary.csv'

#files being read using pandas
time_table = pd.read_csv(time_table_file)
survey = pd.read_csv(survey_file)

#creating labelEncoder for converting string to numeric value for faster processing
le = preprocessing.LabelEncoder()

#timetable ... extracting all the colums data from the csv file
timetable_day = time_table.iloc[:,[0]]
timetable_slot = time_table.iloc[:,[1]]
timetable_subject = time_table.iloc[:,[3]]
timetable_teacher = time_table.iloc[:,[4]]

# result = time_table.iloc[[0], [0, 1, 3, 4]]

# convert to np as dataframe cannot be labelled and hence ravel() function must be used
np_timetable_day = timetable_day.to_numpy()
np_timetable_slot = timetable_slot.to_numpy()
np_timetable_subject = timetable_subject.to_numpy()
np_timetable_teacher = timetable_teacher.to_numpy()

#labels using ravel because dataframe cannot be used to create label
l_timetable_day = Label_encoder.fit_transform(np_timetable_day.ravel())
l_timetable_slot = Label_encoder.fit_transform(np_timetable_slot.ravel())
l_timetable_subject = Label_encoder.fit_transform(np_timetable_subject.ravel())
l_timetable_teacher = Label_encoder.fit_transform(np_timetable_teacher.ravel())


#survey ... extracting all the colums data from the csv file
survey_date = survey.iloc[:,[0]]
survey_day = survey.iloc[:,[2]]
survey_slot = survey.iloc[:,[3]]
survey_class = survey.iloc[:,[4]]
survey_subject = survey.iloc[:,[5]]
survey_teacher = survey.iloc[:,[6]]
survey_response = survey.iloc[:,[8]]
survey_t_score = survey.iloc[:,[9]]
survey_s_score = survey.iloc[:,[10]]
survey_score = survey.iloc[:,[11]]

# convert to np as dataframe cannot be labelled and hence ravel() function must be used
np_survey_date = survey_date.to_numpy()
np_survey_day = survey_day.to_numpy()
np_survey_slot = survey_slot.to_numpy()
np_survey_class = survey_class.to_numpy()
np_survey_subject = survey_subject.to_numpy()
np_survey_teacher = survey_teacher.to_numpy()
np_survey_response = survey_response.to_numpy()
np_survey_t_score = survey_t_score.to_numpy()
np_survey_s_score = survey_s_score.to_numpy()
np_survey_score = survey_score.to_numpy()



#labels using ravel because dataframe cannot be used to create label
l_survey_date = Label_encoder.fit_transform(np_survey_date.ravel())
l_survey_day = Label_encoder.fit_transform(np_survey_day.ravel())
l_survey_slot = Label_encoder.fit_transform(np_survey_slot.ravel())
l_survey_class = Label_encoder.fit_transform(np_survey_class.ravel())
l_survey_subject = Label_encoder.fit_transform(np_survey_subject.ravel())
l_survey_teacher = Label_encoder.fit_transform(np_survey_teacher.ravel())
l_survey_response = Label_encoder.fit_transform(np_survey_response.ravel())
l_survey_t_score = Label_encoder.fit_transform(np_survey_t_score.ravel())
l_survey_s_score = Label_encoder.fit_transform(np_survey_s_score.ravel())
l_survey_score = Label_encoder.fit_transform(np_survey_score.ravel())

# Combine all the features
# features = list(zip(l_survey_date, l_survey_day, l_survey_slot, l_survey_class, l_survey_subject, l_survey_teacher, l_survey_response, l_survey_t_score, l_survey_s_score))
features = list(zip(l_survey_day, l_survey_slot, l_survey_subject, l_survey_teacher))

# Machine learning implementation
model = KNeighborsClassifier(n_neighbors = 3)

model.fit(features, np_survey_score.ravel())

result = 0
iter = len(time_table)
print(iter)
# A loop for going through the 
for i in range (0,iter):
    predicted = model.predict([[l_timetable_day[i],l_timetable_slot[i],l_timetable_subject[i],l_timetable_teacher[i]]])
    result = result + predicted
    print('.')

result = result/iter
print('This time table has a score of: ',result)

# printing all the read data for debugging
# print(time_table)
# print(survey)
