import pandas as pd

df_test = pd.read_csv(r'C:\Users\Nazanin.Dameshghi\PycharmProjects\FBRA\data\First_Prune_Models_Test.csv', low_memory=False)
df_train = pd.read_csv(r'C:\Users\Nazanin.Dameshghi\PycharmProjects\FBRA\data\First_Prune_Models_Train.csv', low_memory=False)
df = pd.concat([df_train, df_test])

df = df.drop(["MMs_In_202101_RC", "Provider_Count_County", "County_Recipient_Provider_Ratio", "County", "Unnamed: 0"],
             axis=1)

X = df.drop('PMPM_Cost', axis=1)
y = df[['PMPM_Cost']]

# Feature Engineering
categorical_features = []
ordinal_features = []
features = list(X.columns)
for feature in features:
    number_of_values = len(X[feature].unique())
    if number_of_values <= 2:
        categorical_features.append(feature)
    else:
        ordinal_features.append(feature)

categorical_features.append('Race')
ordinal_features.remove('Race')

ordinal_features_dict = {
    "A13A_Levelled": ['Lives with Dependents', 'Lives with Support', 'Lives Alone'],
    "C1_Levelled": ['Independent', 'Modified Independence', 'Minimally Impaired', 'Moderately Impaired',
                    'Severely Impaired'],
    "D1_Levelled": ['Understood', 'Usually Understood', 'Sometimes Understood', 'Rarely Understood'],
    "D2_Levelled": ['Understands', 'Usually Understands', 'Sometimes Understands', 'Rarely Understands'],
    "G1AP_Levelled": ['Independent', 'Supervision', 'Assistance', 'Mostly or Completely Dependent',
                      'Activity Did Not Occur'],
    "G1BP_Levelled": ['Limited To No Assistance', 'Assistance',
                      'Mostly or Completely Dependent or Activity Did Not Occur'],
    "G1CP_Levelled": ['Independent', 'Limited Assistance', 'Assistance',
                      'Mostly or Completely Dependent or Activity Did Not Occur'],
    "G1DP_Levelled": ['Independent/Setup Help', 'Limited Assistance', 'Completely Dependent',
                      'Extensive Assistance/Activity Did Not Occur'],
    "G1EP_Levelled": ['Independent', 'Assistance', 'Completely Dependent', 'Activity Did Not Occur'],
    "G1FP_Levelled": ['Independent/Assistance', 'Extensive Assistance', 'Completely Dependent',
                      'Activity Did Not Occur'],
    "G1GP_Levelled": ['Independent/Supervision', 'Limited/Extensive Assistance',
                      'Maximal Assistance/Activity Did Not Occur', 'Completely Dependent'],
    "G1HP_Levelled": ['Independent/Setup', 'Supervision/Limited Assistance', 'Extensive Assistance',
                      'Maximal Assistance/Completely Dependent', 'Activity Did Not Occur'],
    "G2A_Levelled": ['Independent/Setup', 'Supervision', 'Minimal Assistance/Activity Did Not Occur',
                     'Extensive Assistance', 'Maximal Assistance', 'Complete Dependence'],
    "G2B_Levelled": ['Independent', 'Setup', 'Supervision/Minimal Assistance/Activity Did Not Occur',
                     'Extensive Assistance', 'Maximal Assistance/Complete Dependence'],
    "G2C_Levelled": ['Independent', 'Setup', 'Supervision', 'Minimal Assistance',
                     'Extensive Assistance/Activity Did Not Occur', 'Maximal Assistance', 'Complete Dependence'],
    "G2D_Levelled": ['Independent/Setup', 'Supervision', 'Minimal Assistance/Activity Did Not Occur',
                     'Extensive Assistance', 'Maximal Assistance/Complete Dependence'],
    "G2E_Levelled": ['Independent/Setup/Supervision', 'Minimal Assistance', 'Extensive Assistance',
                     'Maximal Assistance/Complete Dependence/Activity Did Not Occur'],
    "G2F_Levelled": ['Independent/Setup/Supervision', 'Minimal Assistance',
                     'Extensive Assistance/Maximal Assistance/Activity Did Not Occur', 'Complete Dependence'],
    "G2G_Levelled": ['Independent/Setup', 'Supervision', 'Minimal Assistance', 'Extensive Assistance',
                     'Maximal Assistance/Complete Dependence/Activity Did Not Occur'],
    "G2H_Levelled": ['Independent/Setup', 'Supervision', 'Minimal Assistance',
                     'Extensive Assistance/Activity Did Not Occur', 'Maximal Assistance/Complete Dependence'],
    "G2J_Levelled": ['Independent/Setup', 'Supervision', 'Minimal Assistance/Activity Did Not Occur',
                     'Extensive/Maximal Assistance/Complete Dependence'],
    "G2K_Levelled": ['Independent/Setup/Supervision', 'Minimal Assistance',
                     'Extensive Assistance/Activity Did Not Occur', 'Maximal Assistance/Complete Dependence'],
    "J1_Levelled": ['No Falls Ever', '0 or 1 Falls In Last 3 Days', '2 Falls In Last 3 Days'],
    "J6A_Levelled": ['No Pain Ever', 'Pain On Less Than 2 Days In Last 3 Days', 'Pain Daily In Last 3 Days'],
    "J6B_Levelled": ['No Pain Ever', 'Mild/Moderate Pain', 'Severe/Excruciating Pain'],
    "J6E_Levelled": ['No Pain Ever', 'Pain - At Least Somewhat Controlled', 'Uncontrolled'],
    "K3_Levelled": ['Normal Food Intake', 'Modified Independence', 'Assistance/Tube Feeding'],
    "ltss_mm_Levelled": ['1 Year or Less', 'Between 1 and 2 Years', 'Between 2 and 3 Years', 'More Then 3 Years']
}
ordinal_transformer = {}
for column in ordinal_features_dict:
    mapper = {}
    ordered_list = ordinal_features_dict[column]
    for idx, distinct_value in enumerate(ordered_list):
        mapper[distinct_value] = idx
    ordinal_transformer[column] = mapper

X_non_ordinal_categories = pd.get_dummies(X[categorical_features], drop_first=True)
X_ord = X[ordinal_features]
X_ordinal_categories = X_ord.replace(ordinal_transformer)
T = pd.concat([X_non_ordinal_categories, X_ordinal_categories], axis=1)
X = T.copy(deep=True)

X.to_csv(r'C:\Users\Nazanin.Dameshghi\PycharmProjects\FBRA\data\features.csv', index=False)
y.to_csv(r'C:\Users\Nazanin.Dameshghi\PycharmProjects\FBRA\data\labels.csv', index=False)
