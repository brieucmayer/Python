
# GROUP 20 : THEO LEMONNIER AND BRIEUC MAYER
# THIS CODE HAS TO BE RUNNED IN A FOLDER CONTAINING ALL THE DATASETS THAT WERE GIVEN FOR THE EXERCICE.
# IT CONTAINS COMMENTS EXPLAINING THE COMPUTATION REALIZED.
# IT PRINTS THE IMPORTANT STATISTICS AND GRAPHS THAT WE ADDED IN THE JOINT REPORT WHICH EXPLAINS THE RESULT OBTAINED.


import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


#############################################     PREPARATION OF THE DATA    ###########################################



#We import our principal datasets before making manipulation 
dataset = pd.read_csv('dataset.csv')
dt_club = pd.read_csv('dataset_CLUB.csv')
dt_contract = pd.read_csv('dataset_contract.csv')

#We observe our principal datasets to understand what they contain.
#We understand that we will be able to merge our data with the KEY column dataset.
#We keep all the dataset individuals, even if we do not have information about their club or their contact.
#If we do not specify how = left, the merging would only keep 774 individuals out of the 10000.
#Those would correspond to the individuals who have a club and a contract specified, which is not required.
dataset = dataset.merge(dt_club, how = 'left')
del dt_club
dataset = dataset.merge(dt_contract, how = 'left')
del dt_contract


#We change the outcome variable to a boolean one.
dataset = dataset.rename(columns={'OUTCOME': 'Success'})
dataset = dataset.join(pd.get_dummies(dataset['Success']))
dataset['Success'] = dataset.Success =='success'

#To understand more directly our results, we replace the INSEE code by the real description of the variables.
#We always merge with how = left, to keep our 10000 individuals.
#Like we explained before even with lacking information, every individual could add other valuable information.

#club
club = pd.read_csv('code_CLUB.csv')
dataset = dataset.rename(columns={'CLUB': 'code_club'})
club = club.rename(columns={'code': 'code_club'})
dataset = dataset.merge(club, how='left', on='code_club').drop('content',axis=1)
dataset = dataset.rename(columns={'description': 'club'})
dataset = dataset.drop('code_club', axis=1)

#contrat
contract = pd.read_csv('code_contract.csv')
dataset = dataset.rename(columns={'contract': 'code_contract'})
contract = contract.rename(columns={'code': 'code_contract'})
dataset = dataset.merge(contract, how='left', on='code_contract').drop('content',axis=1)
dataset = dataset.rename(columns={'description': 'contract'})
dataset = dataset.drop('code_contract', axis=1)

#occupation 24
occu24 = pd.read_csv('code_Occupation_24.csv')
dataset = dataset.rename(columns={'Occupation_24': 'code_occu24'})
occu24 = occu24.rename(columns={'code': 'code_occu24'})
dataset = dataset.merge(occu24, how='left', on='code_occu24').drop('content',axis=1)
dataset = dataset.rename(columns={'description': 'occu24'})
dataset = dataset.drop('code_occu24', axis=1)

#occu8
occu8 = pd.read_csv('code_occupation_8.csv')
dataset = dataset.rename(columns={'occupation_8': 'code_occu8'})
occu8 = occu8.rename(columns={'code': 'code_occu8'})
dataset = dataset.merge(occu8, how='left', on='code_occu8').drop('content',axis=1)
dataset = dataset.rename(columns={'description': 'occu8'})
dataset = dataset.drop('code_occu8', axis=1)

#household 
household = pd.read_csv('code_household.csv')
dataset = dataset.rename(columns={'household': 'code_household'})
household = household.rename(columns={'code': 'code_household'})
dataset = dataset.merge(household, how='left', on='code_household').drop('content',axis=1)
dataset = dataset.rename(columns={'description': 'household'})
dataset = dataset.drop('code_household', axis=1)

#education 
education = pd.read_csv('code_Highest_Degree.csv')
dataset = dataset.rename(columns={'Highest_degree': 'code_education'})
education = education.rename(columns={'code': 'code_education'})
dataset = dataset.merge(education, how='left', on='code_education').drop('content',axis=1)
dataset = dataset.rename(columns={'description': 'education'})
dataset = dataset.drop('code_education', axis=1)

#act 
act = pd.read_csv('code_ACT.csv')
dataset = dataset.rename(columns={'ACT': 'code_act'})
act = act.rename(columns={'code': 'code_act'})
dataset = dataset.merge(act, how='left', on='code_act').drop('content',axis=1)
dataset = dataset.rename(columns={'description': 'act'})
dataset = dataset.drop('code_act', axis=1)

#We define each categorical variable as such
for var in ['club','sex','occu24','contract','occu8','household','education','act']:
    dataset.var = pd.Categorical(dataset.var)

dataset = dataset.join(pd.get_dummies(dataset['sex']))

#We import the datasets needed for the geographical part.
cities  = pd.read_csv('city_adm.csv')
cityloc = pd.read_csv('city_loc.csv')
citypop = pd.read_csv('city_pop.csv')
dep     = pd.read_csv('departments.csv')
regions = pd.read_csv('regions.csv')

#We observe those datasets to understand what they contain.
#We merge the dataset to create a global dataset with all information about any city.
#We keep all the cities (and we observe that no city is lost), since they all have an INSEE code,
#and every INSEE code appear in all of the corresponding dataset. We drop REG and DEP codes.
#We chose to keep the Lambert coordinates rather than the long and lat, since they are similar to analyse.
for dset in [cityloc,citypop,dep,regions]:
    cities = cities.merge(dset)
    del dset
cities = cities.drop(['DEP','REG', 'Lat', 'Long'], axis=1)

#We define each categorical variable as such
for var in ['nom du département','nom de la région','town_type']:
    cities.var = pd.Categorical(cities.var)

#To be able to analyse, we merge both dataframes
#We also drop the colomns Insee, that we will not use anymore, to simplify our dataframe.
dataset = dataset.merge(cities).drop(['Insee'], axis=1)

#We can now check our dataset
print(dataset.head())









###############################################       PRIMARY DATA EXPLORATION     ############################################





#GENDER
#Nbr of men/female divided by number of individuals
gender = dataset.groupby(['sex']).size()/dataset['sex'].count()
print('\n The gender repartition is : \n', gender)
#Scatterplot individuals with their position thanks to Lambert coordinates, differenciated by their gender with color
plt.figure()
g_1 = sns.scatterplot(data = dataset, x = 'X', y='Y', hue = 'sex', s = 2)
g_1.set_title("Gender repartition in our dataset")




#AGE
#Start with printing usual statistics
print('\n The average age is : ', dataset['current_age'].mean())
print('\n The first quartile age is : ', dataset['current_age'].quantile(0.25))
print('\n The median age is : ', dataset['current_age'].median())
print('\n The third quartile age is : ', dataset['current_age'].quantile(0.75))
#Cut the ages in ranges of 10 to obtain grouped statistics
dataset['agecut'] = pd.cut(dataset['current_age'], range(10,111,10))
#Number of people in each group / number of individuals
age = dataset.groupby(['agecut']).size()/dataset['agecut'].count()
print('\n The age repartition is : \n', age)
#Scatterplot for each gender the repartition of the ages of the individuals
plt.figure()
g_21 = sns.catplot(x="sex", y="current_age", kind="violin", split=True, data=dataset)
g_21.fig.suptitle("Age repartition in our dataset")
#Scatterplot for each age the position of the individuals, differenciated by their gender with color
plt.figure()
g_2 = sns.FacetGrid(dataset, col="agecut", col_wrap = 4)
g_2.map(sns.scatterplot, "X", "Y", s = 15)
plt.subplots_adjust(top=0.9)
g_2.fig.suptitle("Area repartition for each age range in our dataset")




#HIGHEST DEGREE
#For the following variable, we compute the repartition by grouping, obtaining sizes of groupe and dividing by the total number of individuals
educ = dataset.groupby(['education']).size().sort_values(ascending=False)/dataset['education'].count()
print('\n The highest degree obtained repartition is : \n', educ)




#STUDENTS
#Count people which are studying and divide by the number of people
stud = dataset['Studying'][dataset.Studying].count()/dataset['Studying'].count()
print('\n The studying rate is : ', stud)




#HOUSEHOLD
#Same computation of the repartition
househ = dataset.groupby(['household']).size().sort_values(ascending=False)/dataset['household'].count()
print('\n The household type repartition is : \n', househ)
#Ploting the repartition
plt.figure()
g_22 = sns.catplot(data = dataset, x = 'household', kind ='count')
g_22.set_xticklabels(rotation=90).fig.suptitle('Household type repartition in our set')




#ACTIVITY
#Same computation
act = dataset.groupby(['act']).size().sort_values(ascending=False)/dataset['act'].count()
print('\n The activity repartition is : \n', act)
#Same ploting
plt.figure()
g_4 = sns.catplot(data = dataset, y = 'act', kind ='count', hue ='sex', order = dataset['act'].value_counts().index)
g_4.fig.suptitle('Activity repartition in our set')




#OCCUPATION
#Same computation
region = dataset.groupby(['occu8']).size().sort_values(ascending=False)/dataset['occu8'].count()
print('\n The repartition of the people depending on their occupation is : \n', region)
#Same ploting
plt.figure()
g_41 = sns.catplot(data = dataset, y = 'occu8', kind ='count', order = dataset['occu8'].value_counts().index)
g_41.set_xticklabels(rotation=90).fig.suptitle('Occupation repartition in our set')




#CITY POPULATION
#Usual statistics
print('\n The average city population of people of this dataframe is : ', dataset['inhabitants'].mean())
print('\n The first quartile city population is : ', dataset['inhabitants'].quantile(0.25))
print('\n The median city population is : ', dataset['inhabitants'].quantile(0.5))
print('\n The third quartile city population is : ', dataset['inhabitants'].quantile(0.75))
#Cuts in ranges and repartition
dataset['inhabcuts'] = pd.cut(dataset['inhabitants'], range(0,500001,50000))
cityrep = dataset.groupby(['inhabcuts']).size()/dataset['inhabcuts'].count()
print('\n The cities population repartition is : \n', cityrep)
#Repartition ploting
plt.figure()
g_44 = sns.catplot(data = dataset, x = 'inhabcuts', kind ='count')
g_44.set_xticklabels(rotation=90).fig.suptitle('Population of the cities or residence repartition in our set')




#REGIONS
#Same repartition computation
region = dataset.groupby(['Nom de la région']).size().sort_values(ascending=False)/dataset['Nom de la région'].count()
print('\n The repartition of the people in "régions" is : \n', region)
#Ploting the repartition
plt.figure()
g_42 = sns.catplot(data = dataset, x = 'Nom de la région', kind ='count', order = dataset['Nom de la région'].value_counts().index)
g_42.set_xticklabels(rotation=90).fig.suptitle('Regional repartition in our set')
#Ploting in a geographical way
plt.figure()
g_43 = sns.scatterplot(data = dataset, x = 'X', y='Y', alpha=.7, hue = 'Nom de la région', s = 7)
g_43.legend(loc='center left', bbox_to_anchor=(1, 0.5))
g_43.set_title("Regional repartition of our population")



#CLUBS
#Computing the rate of registered people
print('\n The rate of people registred in clubs is : ', dataset['club'].notnull().mean())



#OUTCOME
#Computing the success rate
succ = dataset['Success'][dataset.Success].count()/dataset['Success'].count()
print('\n The success rate is : ', succ)
#Ploting the success rate
plt.figure()
g_61 = sns.catplot(data = dataset, x = 'Success', kind ='count')







##################################     PERSONNAL LEVEL ANALYSIS - VARIABLE PER VARIABLE       ####################################################





######## GENDER

#Descriptive point of view : compute the repartition of gender by success group
gender2 = dataset.groupby(['sex','Success']).size()/dataset.groupby(['Success']).size()
gender2 = gender2.unstack()
print('\n The gender repartition in the success and failure set is : \n', gender2)
#Plot a graph of this repartition
plt.figure()
g_6 = sns.catplot(data = dataset, x = 'Success', kind ='count', hue = 'sex')
g_6.set_xticklabels(rotation=90).fig.suptitle('Gender repartition in Success and Failure sets')
#Predictive point of view : compute the repartition of success by gender group
gender3 = dataset.groupby(['Success','sex']).size()/dataset.groupby(['sex']).size()
gender3 = gender3.unstack()
print('\n The success rates in the men and women population are : \n', gender3)
#Ploting it
plt.figure()
g_7 = sns.catplot(data = dataset, x = 'sex', kind ='count', hue = 'Success')
g_7.set_xticklabels(rotation=90).fig.suptitle('Success repartition depending on gender')





######### AGE

#Descriptive point of view : most usual statistics and repartition
agemean = dataset.groupby(['Success'])['current_age'].mean()
agefirstq = dataset.groupby(['Success'])['current_age'].quantile(0.25)
agemedian = dataset.groupby(['Success'])['current_age'].quantile(0.5)
agethirdq = dataset.groupby(['Success'])['current_age'].quantile(0.75)
print('The mean of ages in both sets are : ', agemean)
print('The first quartile of ages in both sets are : ', agefirstq)
print('The median of ages in both sets are : ', agemedian)
print('The third quartile of ages in both sets are : ', agethirdq)
age2 = dataset.groupby(['agecut','Success']).size()/dataset.groupby(['Success']).size()
age2 = age2.unstack()
print('\n The age repartition in the success and failure set is : \n',age2)
#We plot it in a violon way
plt.figure()
g_8 = sns.catplot(y="current_age", x = 'Success', kind="violin", split=True, data=dataset)
g_8.fig.suptitle("Age repartition for Failure and Success sets")
#Predictive point of view : also computing the repartition
age3 = dataset.groupby(['Success','agecut']).size()/dataset.groupby(['agecut']).size()
age3 = age3.unstack()
print('\n The success rates in the different ranges of ages are : \n',age3)
#Ploting this repartition
plt.figure()
g_9 = age3
g_9.columns = g_9.columns.astype(str)
g_9['Success'] = g_9.index
g_9 = g_9.loc[g_9['Success']]
g_9 = g_9.melt(value_vars= g_9.columns[:-1], id_vars = 'Success')
g_9 = g_9.rename(columns={'value':'Rate of success','agecut':'Age Range'})
sns.set_style("whitegrid")
g_9 = sns.relplot(data = g_9, x = 'Age Range', y='Rate of success', hue='Success',legend='full', kind='line')
g_9.set_xticklabels(rotation=90).fig.suptitle('Rate of success at different ages')





######## REGION

#Descriptive point of view : the same kind of methods are now used for every variable
reg2 = dataset.groupby(['Nom de la région','Success']).size()/dataset.groupby(['Success']).size()
reg2 = reg2.unstack()
print('\n The regional repartition in the success and failure set is : \n', reg2)
plt.figure()
graph4ds = sns.scatterplot(data = dataset, x = 'X',y='Y', hue = 'Success')
#Predictive point of view : the same kind of methods are now used for every variable
reg3 = dataset.groupby(['Success','Nom de la région']).size()/dataset.groupby(['Nom de la région']).size()
reg3 = reg3.unstack().sort_values(True,ascending=False, axis=1)
print('\n The success rates depending on the region are : \n',reg3)
#Ploting
plt.figure()
g = sns.FacetGrid(dataset, col="agecut", hue="Success", col_wrap = 4)
g.map(sns.scatterplot, "X", "Y", alpha=.7)
g.add_legend()
#Ploting
plt.figure()
graph1ds = dataset.groupby(['Success','Nom de la région']).size()/dataset.groupby(['Nom de la région']).size()
graph1ds = graph1ds.reset_index(level=['Success','Nom de la région'])
graph1ds = graph1ds.loc[graph1ds['Success']]
graph1ds = graph1ds.rename(columns={0:'Rate of success'})
graph1ds = graph1ds.sort_values('Rate of success',ascending=False)
sns.set_style("whitegrid")
graph1 = sns.catplot(data=graph1ds, x = 'Nom de la région', y = 'Rate of success')
graph1.set_xticklabels(rotation=90).set(title='Rate of success per Région')
#Define Atlantic coast as all the point below a certain distance from a point in the Atlantic.
Center  = [200000,6200000]
Radius = 650000
conditions = [(dataset['X']-Center[0])**2 + (dataset['Y']-Center[1])**2 <= Radius**2,
              (dataset['X']-Center[0])**2 + (dataset['Y']-Center[1])**2 > Radius**2  ]
choices = ['Atlantic Coast','Rest of France']
dataset['Area'] = np.select(conditions, choices)
#Plot how we divided France
plt.figure()
sns.set(font_scale=1)
g_100 = sns.FacetGrid(dataset, row="Success", hue="Area")
g_100.map(sns.scatterplot, "X", "Y", s = 10)
#Plot the results we obstain
print(dataset.groupby(['Area','Success']).size()/dataset.groupby(['Area']).size())





######## CITYPOP

#Descriptive point of view
habmean = dataset.groupby(['Success'])['inhabitants'].mean()
habfirstq = dataset.groupby(['Success'])['inhabitants'].quantile(0.25)
habmedian = dataset.groupby(['Success'])['inhabitants'].quantile(0.5)
habthirdq = dataset.groupby(['Success'])['inhabitants'].quantile(0.75)
print('The mean of the population of the city of the individuals in both sets are : ', habmean)
print('The first quartile of the population of the city of the individuals in both sets are : ', habfirstq)
print('The median of the population of the city of the individuals in both sets are : ', habmedian)
print('The third quartile of the population of the city of the individuals in both sets are : ', habthirdq)
#By cuts
hab2 = dataset.groupby(['inhabcuts','Success']).size()/dataset.groupby(['Success']).size()
hab2 = hab2.unstack()
print('\n The cities populations repartition in the success and failure set is : \n',hab2)
#Ploting
plt.figure()
g_81 = sns.catplot(y="inhabitants", x = 'Success', kind="violin", split=True, data=dataset)
g_81.fig.suptitle("Population of the city of the individuals repartition for Failure and Success sets")
#Predictive point of view
hab3 = dataset.groupby(['Success','inhabcuts']).size()/dataset.groupby(['inhabcuts']).size()
hab3 = hab3.unstack()
print('\n The success rates in the different ranges of cities populations are : \n',hab3)
#Ploting
plt.figure()
g_83 = hab3
g_83.columns = g_83.columns.astype(str)
g_83['Success'] = g_83.index
g_83 = g_83.loc[g_83['Success']]
g_83 = g_83.melt(value_vars= g_83.columns[:-1], id_vars = 'Success')
g_83 = g_83.rename(columns={'value':'Rate of success','inhabcuts':'City Population Range'})
sns.set_style("whitegrid")
g_83 = sns.relplot(data = g_83, x = 'City Population Range', y='Rate of success', hue='Success',legend='full',s=100)
g_83.set_xticklabels(rotation=90).fig.suptitle('Rate of success for different populations of city')





######## CLUB

#Descriptive point of view
dataset['hasclub']=dataset['club'].notnull().replace()
club2 = dataset.groupby(['hasclub','Success']).size()/dataset.groupby(['Success']).size()
club2 = club2.unstack().rename(index={True: 'Has Club', False:'Has no club'})
print('\n The part of the success and failure sets having clubs are : \n', club2)
#Predictive point of view
club3 = dataset.groupby(['Success','hasclub']).size()/dataset.groupby(['hasclub']).size()
club3 = club3.unstack().rename(columns={True: 'Has Club', False:'Has no club'})
print('\n The success rates depending on having a club or not are : \n', club3)
#To get the following plot, we used the exemple given on the matplotlab open source gallery.
def autolabel(rects, xpos='center'):
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')
dataset['club1'] = dataset['club']
dataset['club1'] = dataset['club1'].fillna(0)
dataset['sport'] = ([1 if c != 0 else 0 for c in dataset['club1']])
gb_success_sex = list(dataset.groupby(['sport','success']))
nb_failure_women = gb_success_sex[0][1].shape[0]
nb_success_women = gb_success_sex[1][1].shape[0]
nb_failure_male = gb_success_sex[2][1].shape[0]
nb_success_male = gb_success_sex[3][1].shape[0]
labels1 = 'Success', 'Failure'
labels2 = 'Sportive', 'No sport', 'Sportive', 'No sport'
fig, ax = plt.subplots(figsize=(10,20))
size = 0.3
vals = np.array([[nb_success_male, nb_success_women], [nb_failure_male, nb_failure_women]])
cmap = plt.get_cmap("tab20c")
outer_colors = cmap(np.arange(3)*4)
inner_colors = cmap(np.array([1, 2, 5, 6]))
ax.pie(vals.sum(axis=1), radius=1, colors=outer_colors,
       wedgeprops=dict(width=size, edgecolor='w'), labels=labels1)
ax.pie(vals.flatten(), radius=1-size, colors=inner_colors,
       wedgeprops=dict(width=size, edgecolor='w'), labels=labels2,autopct='%1.1f%%')
ax.set(aspect="equal", title='Repartition of failure \ success by sport')
plt.show()




######## EDUCATION

#Descriptive point of view
educ2 = dataset.groupby(['education','Success']).size()/dataset.groupby(['Success']).size()
educ2 = educ2.unstack()
print('\n The education repartition in the success and failure set is : \n',educ2)
#Predictive point of view
educ3 = dataset.groupby(['Success','education']).size()/dataset.groupby(['education']).size()
educ3 = educ3.unstack().sort_values(True,ascending=False, axis=1)
print('\n The success rates depending on the household categories are : \n',educ3)
#To get the following plot, we used the exemple given on the matplotlab open source gallery.
gb_success_educ = list(dataset.groupby(['education','success']))
educ_success = []
educ_failure = []
educ_ticklabels = []
for i in range(len(gb_success_educ) // 2) :
    educ_failure.append(gb_success_educ[i*2][1].shape[0])
    educ_success.append(gb_success_educ[i*2 + 1][1].shape[0])
    educ_ticklabels.append(gb_success_educ[i*2][0][0].replace(' ','\n'))
ind = np.arange(len(educ_success))  # the x locations for the groups
width = 0.35  # the width of the bars
fig, ax = plt.subplots(figsize=(15,15))
rects1 = ax.bar(ind - width/2, educ_success, width,label='Success')
rects2 = ax.bar(ind + width/2, educ_failure, width,label='Failure')
ax.set_ylabel('Scores')
ax.set_title('Numbers of success / failure by education')
ax.set_xticks(ind)
ax.set_xticklabels(educ_ticklabels)
ax.legend()
autolabel(rects1, "left")
autolabel(rects2, "right")
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(18)
fig.tight_layout()
plt.show()




######## HOUSEHOLD

#Descriptive point of view
househ2 = dataset.groupby(['household','Success']).size()/dataset.groupby(['Success']).size()
househ2 = househ2.unstack()
print('\n The households repartition in the success and failure set is : \n', househ2)
#Predictive point of view
househ3 = dataset.groupby(['Success','household']).size()/dataset.groupby(['household']).size()
househ3 = househ3.unstack().sort_values(True,ascending=False, axis=1)
print('\n The success rates depending on the household categories are : \n',househ3)
#Plot
nbrhouseh = dataset.groupby(['household']).size()
g_300 = dataset.groupby(['household','Success']).size().unstack('Success')
g_300 = g_300[True]/nbrhouseh
sns.set_style("whitegrid")
g_300 = g_300.rename('Rate of success').sort_values()
g_300 = g_300.to_frame()
g_300['Household type'] = g_300.index
g_300 = sns.relplot(data = g_300, y = 'Household type', x = 'Rate of success',legend='full',s=100)




######## ACTIVITY

#Descriptive point of view
act2 = dataset.groupby(['act','Success']).size()/dataset.groupby(['Success']).size()
act2 = act2.unstack()
print('\n The activity repartition in the success and failure set is : \n', act2)
#Predictive point of view
act3 = dataset.groupby(['Success','act']).size()/dataset.groupby(['act']).size()
act3 = act3.unstack().sort_values(True,ascending=False, axis=1)
print('\n The success rates depending on the activity are : \n',act3)
#To get the following plot, we used the exemple given on the matplotlab open source gallery.
gb_success_act = list(dataset.groupby(['act','success']))
act_success = []
act_failure = []
act_ticklabels = []
for i in range(len(gb_success_act) // 2) :
    act_failure.append(gb_success_act[i*2][1].shape[0])
    act_success.append(gb_success_act[i*2 + 1][1].shape[0])
    act_ticklabels.append(gb_success_act[i*2][0][0].replace(' ','\n'))
ind = np.arange(len(act_success))  # the x locations for the groups
width = 0.35  # the width of the bars
fig, ax = plt.subplots(figsize=(15,15))
rects1 = ax.bar(ind - width/2, act_success, width,label='Success')
rects2 = ax.bar(ind + width/2, act_failure, width,label='Failure')
ax.set_ylabel('Scores')
ax.set_title('Numbers of success / failure by activity')
ax.set_xticks(ind)
ax.set_xticklabels(act_ticklabels)
ax.legend()
autolabel(rects1, "left")
autolabel(rects2, "right")
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(18)
fig.tight_layout()
plt.show()







#######################################################      GROUPED ANALYSIS      ####################################################







######## GROUPED BY GENDER



#AGE EFFECT
g_age_mean   = dataset.groupby(['sex'])['current_age'].mean()
g_age_median  = dataset.groupby(['sex'])['current_age'].median()
g_s_age_mean = dataset.groupby(['sex','Success'])['current_age'].mean().unstack('Success')
g_s_age_median = dataset.groupby(['sex','Success'])['current_age'].median().unstack('Success')
print('The mean age of the success and failure groups depending on the sex is : \n', g_s_age_mean)
print('The median age of the success and failure group depending on the sex is : \n', g_s_age_median)
#Cut
g_agecut = dataset.groupby(['sex','agecut']).size().unstack('agecut')
g_s_agecut = dataset.groupby(['sex','agecut','Success']).size().unstack('Success')
g_s_agecut = g_s_agecut[True].unstack('agecut')/g_agecut
#Plot
plt.figure()
g_102 = g_s_agecut
g_102.columns = g_102.columns.astype(str)
g_102['sex']=g_102.index
g_102 = g_102.melt(value_vars= g_102.columns[:-1], id_vars = 'sex')
g_102 = g_102.rename(columns={'value':'Rate of success','agecut':'Age Range'})
sns.set_style("whitegrid")
g_102 = sns.relplot(data = g_102, x = 'Age Range', y='Rate of success', hue='sex',legend='full', kind='line')
g_102.set_xticklabels(rotation=90)


 
#POPULATION EFFECT
g_citypop_mean = dataset.groupby(['sex'])['inhabitants'].mean()
g_citypop_median = dataset.groupby(['sex'])['inhabitants'].median()
print('The mean population of city of the people depending on the sex is : \n', g_citypop_mean)
print('The median population of city of the people depending on the sex is : \n', g_citypop_median)
g_s_citypop_mean = dataset.groupby(['sex','Success'])['inhabitants'].mean().unstack('Success')
g_s_citypop_median = dataset.groupby(['sex','Success'])['inhabitants'].median().unstack('Success')
print('The mean population of city of the people of the success and failure groups depending on the sex is : \n', g_s_citypop_mean)
print('The median population of city of the people of the success and failure group depending on the sex is : \n', g_s_citypop_median)
#Cut
g_inhabcut = dataset.groupby(['sex','inhabcuts']).size().unstack('inhabcuts')
g_s_inhabcut = dataset.groupby(['sex','inhabcuts','Success']).size().unstack('Success')
g_s_inhabcut = g_s_inhabcut[True].unstack('inhabcuts')/g_inhabcut
#Plot
plt.figure()
g_103 = g_s_inhabcut
g_103.columns = g_103.columns.astype(str)
g_103['sex']=g_103.index
g_103 = g_103.melt(value_vars= g_103.columns[:-1], id_vars = 'sex')
g_103 = g_103.rename(columns={'value':'Rate of success','inhabcuts':'Population Range'})
sns.set_style("whitegrid")
g_103 = sns.relplot(data = g_103, x = 'Population Range', y='Rate of success', hue='sex',legend='full', s = 100)
g_103.set_xticklabels(rotation=90)



#REGIONS EFFECT
r_gender = dataset.groupby(['Nom de la région','sex']).size().unstack('sex')
r_gender_size = dataset.groupby(['Nom de la région','sex','Success']).size().unstack('Success')
r_gender_diff = r_gender_size[True].unstack('sex')/r_gender
r_gender_diff = r_gender_diff.sort_values('Female')
#Plot
plt.figure()
g_111 = r_gender_diff
g_111.columns = g_111.columns.astype(str)
g_111['Nom de la région']=g_111.index
g_111 = g_111.melt(value_vars= g_111.columns[:-1], id_vars = 'Nom de la région')
g_111 = g_111.rename(columns={'value':'Rate of success','sex':'Gender'})
sns.set_style("whitegrid")
g_111 = sns.relplot(data = g_111, x = 'Nom de la région', y='Rate of success', col='Gender',legend='full', col_wrap=4)
g_111.set_xticklabels(rotation=90)








######## GROUPED BY REGION


r = dataset.groupby(['Nom de la région']).size()
r_s_rate = dataset.groupby(['Nom de la région','Success']).size().unstack('Success')
r_s_rate = r_s_rate[True]/r
r_s_rate = r_s_rate.sort_values()
print('\n As a reminder, the success rates depending on the regions are : \n',r_s_rate)



#GENDER EFFECT
r_gender = dataset.groupby(['Nom de la région','sex']).size().unstack('sex')
r_gender_size = dataset.groupby(['Nom de la région','sex','Success']).size().unstack('Success')
r_gender_diff = r_gender_size[True].unstack('sex')/r_gender
r_gender_diff = r_gender_diff.sort_values('Female')
#Plot
plt.figure()
g_112 = r_gender_diff
g_112.columns = g_112.columns.astype(str)
g_112['Nom de la région']=g_112.index
g_112 = g_112.melt(value_vars= g_112.columns[:-1], id_vars = 'Nom de la région')
g_112 = g_112.rename(columns={'value':'Rate of success','sex':'Gender'})
sns.set_style("whitegrid")
g_112 = sns.barplot(data = g_112, x = 'Nom de la région', y='Rate of success', hue = 'Gender')
g_112.set_xticklabels(rotation=90, labels = g_112.get_xticklabels())



#AGE EFFECT
r_age = dataset.groupby(['Nom de la région','agecut']).size().unstack('agecut')
r_age_size = dataset.groupby(['Nom de la région','agecut','Success']).size().unstack('Success')
r_age_diff = r_age_size[True].unstack('agecut')/r_age
#Plot
plt.figure()
sns.set(font_scale=1.5)
graph2ds = r_age_diff
graph2ds.columns = graph2ds.columns.astype(str)
graph2ds['Nom de la région']=graph2ds.index
graph2ds = graph2ds.melt(value_vars= graph2ds.columns[:-1], id_vars = 'Nom de la région')
graph2ds = graph2ds.rename(columns={'value':'Rate of success','agecut':'Age Range'})
sns.set_style("whitegrid")
graph2 = sns.relplot(data = graph2ds, x = 'Age Range', y='Rate of success', col='Nom de la région',legend='full', col_wrap=4, kind='line')
graph2.set_xticklabels(rotation=90)



############################ A CONCLUSION COMPUTATION ########################################

#We only keep men on the Atlantic coast
final = dataset[(dataset['sex']=='Male') & (dataset['Area'] == 'Atlantic Coast')]
print('There are ',final['sex'].size, ' men in the Atlantic Coast region we build.')
plt.figure()
sns.set(font_scale=1)
f_age = final.groupby(['agecut','Success']).size()/final.groupby(['agecut']).size()
f_age = f_age.filter(like = 'True', axis=0)
f_age = f_age.droplevel(1)
f_age = f_age.to_frame()
f_age = f_age.rename(columns={0:'Rate of success'})
g_1000 = sns.barplot(data = f_age, x = f_age.index, y ='Rate of success')
g_1000.set_xticklabels(rotation=90, labels = g_1000.get_xticklabels())
g_1000.set_title('Rates of success for men in Atlantic Coast region depending on their age')










