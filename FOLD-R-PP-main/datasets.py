from foldrpp import Classifier


def acute():
    attrs = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    nums = ['a1']
    model = Classifier(attrs=attrs, numeric=nums, label='label', pos='yes')
    data = model.load_data('data/acute/acute.csv')
    print('\n% acute dataset', len(data), len(data[0]))
    return model, data


def adult():
    attrs = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
    'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']
    nums = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    model = Classifier(attrs=attrs, numeric=nums, label='label', pos='<=50K')
    data = model.load_data('data/adult/adult.csv')
    print('\n% adult dataset', len(data), len(data[0]))
    return model, data


def autism():
    attrs = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'age', 'gender', 'ethnicity', 'jaundice',
             'pdd', 'used_app_before', 'relation']
    nums = ['age']
    model = Classifier(attrs=attrs, numeric=nums, label='label', pos='NO')
    data = model.load_data('data/autism/autism.csv')
    print('\n% autism dataset', len(data), len(data[0]))
    return model, data


def breastw():
    attrs = ['clump_thickness', 'cell_size_uniformity', 'cell_shape_uniformity', 'marginal_adhesion',
    'single_epi_cell_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses']
    nums = attrs
    model = Classifier(attrs=attrs, numeric=nums, label='label', pos='benign')
    data = model.load_data('data/breastw/breastw.csv')
    print('\n% breastw dataset', len(data), len(data[0]))
    return model, data


def cars():
    attrs = ['buying', 'maint', 'doors', 'persons', 'lugboot', 'safety']
    model = Classifier(attrs=attrs, numeric=[], label='label', pos='negative')
    data = model.load_data('data/cars/cars.csv')
    print('\n% cars dataset', len(data), len(data[0]))
    return model, data


def credit():
    attrs = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15']
    nums = ['a2', 'a3', 'a8', 'a11', 'a14', 'a15']
    model = Classifier(attrs=attrs, numeric=nums, label='label', pos='-')
    data = model.load_data('data/credit/credit.csv')
    print('\n% credit dataset', len(data), len(data[0]))
    return model, data


def heart():
    attrs = ['age', 'sex', 'chest_pain', 'blood_pressure', 'serum_cholestoral', 'fasting_blood_sugar',
    'resting_electrocardiographic_results', 'maximum_heart_rate_achieved', 'exercise_induced_angina', 'oldpeak',
    'slope', 'major_vessels', 'thal']
    nums = ['age', 'blood_pressure', 'serum_cholestoral', 'maximum_heart_rate_achieved', 'oldpeak']
    model = Classifier(attrs=attrs, numeric=nums, label='label', pos='absent')
    data = model.load_data('data/heart/heart.csv')
    print('\n% heart dataset', len(data), len(data[0]))
    return model, data


def kidney():
    attrs = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv',
    'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    nums = ['age', 'bp', 'sg', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']
    model = Classifier(attrs=attrs, numeric=nums, label='label', pos='ckd')
    data = model.load_data('data/kidney/kidney.csv')
    print('\n% kidney dataset', len(data), len(data[0]))
    return model, data


def krkp():
    attrs = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16',
    'a17', 'a18', 'a19', 'a20', 'a21', 'a22', 'a23', 'a24', 'a25', 'a26', 'a27', 'a28', 'a29', 'a30', 'a31', 'a32',
    'a33', 'a34', 'a35', 'a36']
    model = Classifier(attrs=attrs, numeric=[], label='label', pos='won')
    data = model.load_data('data/krkp/krkp.csv')
    print('\n% krkp dataset', len(data), len(data[0]))
    return model, data


def mushroom():
    attrs = ['cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor', 'gill_attachment', 'gill_spacing',
    'gill_size', 'gill_color', 'stalk_shape', 'stalk_root', 'stalk_surface_above_ring', 'stalk_surface_below_ring',
    'stalk_color_above_ring', 'stalk_color_below_ring', 'veil_type', 'veil_color', 'ring_number', 'ring_type',
    'spore_print_color', 'population', 'habitat']
    model = Classifier(attrs=attrs, numeric=[], label='label', pos='p')
    data = model.load_data('data/mushroom/mushroom.csv')
    print('\n% mushroom dataset', len(data), len(data[0]))
    return model, data


def sonar():
    attrs = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16',
    'a17', 'a18', 'a19', 'a20', 'a21', 'a22', 'a23', 'a24', 'a25', 'a26', 'a27', 'a28', 'a29', 'a30', 'a31', 'a32',
    'a33', 'a34', 'a35', 'a36', 'a37', 'a38', 'a39', 'a40', 'a41', 'a42', 'a43', 'a44', 'a45', 'a46', 'a47', 'a48',
    'a49', 'a50', 'a51', 'a52', 'a53', 'a54', 'a55', 'a56', 'a57', 'a58', 'a59', 'a60']
    nums = attrs
    model = Classifier(attrs=attrs, numeric=nums, label='label', pos='Mine')
    data = model.load_data('data/sonar/sonar.csv')
    print('\n% sonar dataset', len(data), len(data[0]))
    return model, data


def voting():
    attrs = ['handicapped_infants', 'water_project_cost_sharing', 'budget_resolution', 'physician_fee_freeze',
    'el_salvador_aid', 'religious_groups_in_schools', 'anti_satellite_test_ban', 'aid_to_nicaraguan_contras',
    'mx_missile', 'immigration', 'synfuels_corporation_cutback', 'education_spending', 'superfund_right_to_sue',
    'crime', 'duty_free_exports', 'export_administration_act_south_africa']
    model = Classifier(attrs=attrs, numeric=[], label='label', pos='republican')
    data = model.load_data('data/voting/voting.csv')
    print('\n% voting dataset', len(data), len(data[0]))
    return model, data


def ecoli():
    attrs = ['sn','mcg','gvh','lip','chg','aac','alm1','alm2']
    nums = ['mcg','gvh','lip','chg','aac','alm1','alm2']
    model = Classifier(attrs=attrs, numeric=nums, label='label', pos='cp')
    data = model.load_data('data/ecoli/ecoli.csv')
    print('\n% ecoli dataset', len(data), len(data[0]))
    return model, data


def ionosphere():
    attrs = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15', 'c16',
    'c17', 'c18', 'c19', 'c20', 'c21', 'c22', 'c23', 'c24', 'c25', 'c26', 'c27', 'c28', 'c29', 'c30', 'c31', 'c32',
    'c33', 'c34']
    model = Classifier(attrs=attrs, numeric=attrs, label='label', pos='g')
    data = model.load_data('data/ionosphere/ionosphere.csv')
    print('\n% ionosphere dataset', len(data), len(data[0]))
    return model, data


def wine():
    attrs = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'tot_phenols', 'flavanoids',
    'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'OD_of_diluted', 'proline']
    model = Classifier(attrs=attrs, numeric=attrs, label='label', pos='3')
    data = model.load_data('data/wine/wine.csv')
    print('\n% wine dataset', len(data), len(data[0]))
    return model, data


def credit_card():
    attrs = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
    'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    nums = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    model = Classifier(attrs=attrs, numeric=nums, label='DEFAULT_PAYMENT', pos='0')
    data = model.load_data('data/credit_card/credit_card.csv')
    print('\n% credit card dataset', len(data), len(data[0]))
    return model, data


def rain():
    attrs = ['Month', 'Day', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir',
    'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
    'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday']
    nums = ['Month', 'Day', 'MinTemp', 'MaxTemp', 'Rainfall', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am',
    'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm']
    model = Classifier(attrs=attrs, numeric=nums, label='RainTomorrow', pos='No')
    data = model.load_data('data/rain/rain.csv')
    print('\n% rain dataset', len(data), len(data[0]))
    return model, data


def heloc():
    attrs = ['ExternalRiskEstimate', 'MSinceOldestTradeOpen', 'MSinceMostRecentTradeOpen', 'AverageMInFile',
    'NumSatisfactoryTrades', 'NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec', 'PercentTradesNeverDelq',
    'MSinceMostRecentDelq', 'MaxDelq2PublicRecLast12M', 'MaxDelqEver', 'NumTotalTrades', 'NumTradesOpeninLast12M',
    'PercentInstallTrades', 'MSinceMostRecentInqexcl7days', 'NumInqLast6M', 'NumInqLast6Mexcl7days',
    'NetFractionRevolvingBurden', 'NetFractionInstallBurden', 'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance',
    'NumBank2NatlTradesWHighUtilization', 'PercentTradesWBalance']
    nums = attrs
    model = Classifier(attrs=attrs, numeric=nums, label='RiskPerformance', pos='Good')
    data = model.load_data('data/heloc/heloc_dataset_v1.csv')
    print('\n% heloc dataset', len(data), len(data[0]))
    return model, data


def titanic():
    attrs = ['Sex', 'Age', 'Number_of_Siblings_Spouses', 'Number_Of_Parents_Children', 'Fare', 'Class', 'Embarked']
    nums = ['Age', 'Number_of_Siblings_Spouses', 'Number_Of_Parents_Children', 'Fare']
    model = Classifier(attrs=attrs, numeric=nums, label='Survived', pos='0')
    data_train = model.load_data('data/titanic/train.csv')
    data_test = model.load_data('data/titanic/test.csv')
    print('\n% titanic train dataset', len(data_train), len(data_train[0]))
    print('% titanic test dataset', len(data_test), len(data_test[0]))
    return model, data_train, data_test

#########################################################################################################################

# Following are the datasets that are tried and rules are generated using fOLD - R++

def drug():
    attrs = ['Age', 'Na_to_K', 'Sex', 'BP', 'Cholestrol']
    nums = ['Age', 'Na_to_K']
    model = Classifier(attrs=attrs, numeric=nums, label='Drug', pos='DrugY')
    data = model.load_data('data/drug/drug_dataset.csv')
    print('\n% Drug dataset', len(data), len(data[0]))
    return model, data

def success():
    attrs = ['age', 'interest']
    nums = attrs
    model = Classifier(attrs=attrs, numeric=nums, label='success', pos='1.0')
    data = model.load_data('data/success/success_dataset.csv')
    print('\n% Success dataset', len(data), len(data[0]))
    return model, data

def stars():
    attrs = ['Vmag', 'Plx', 'e_Plx', 'B-V', 'SpType', 'Amag']
    nums = ['Vmag', 'Plx', 'e_Plx', 'B-V', 'Amag']
    model = Classifier(attrs=attrs, numeric=nums, label='TargetClass', pos='0')
    data = model.load_data('data/stars/stars_dataset.csv')
    print('\n% Stars dataset', len(data), len(data[0]))
    return model, data

def bmi():
    attrs = ['Gender', 'Height', 'Weight']
    nums = ['Height', 'Weight']
    model = Classifier(attrs=attrs, numeric=nums, label='Index', pos='5')
    data = model.load_data('data/bmi/bmi_dataset.csv')
    print('\n% BMI dataset', len(data), len(data[0]))
    return model, data

def iris():
    attrs = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    nums = attrs
    model = Classifier(attrs=attrs, numeric=nums, label='species', pos='Iris-setosa')
    data = model.load_data('data/iris/iris_dataset.csv')
    print('\n% Iris dataset', len(data), len(data[0]))
    return model, data

def employee():
    attrs = ['Education', 'JoiningYear', 'City', 'PaymentTier', 'Age',
             'Gender', 'EverBenched', 'ExperienceInCurrentDomain']
    nums = ['JoiningYear', 'PaymentTier', 'Age',
            'ExperienceInCurrentDomain']
    model = Classifier(attrs=attrs, numeric=nums, label='LeaveOrNot', pos='0')
    data = model.load_data('data/employee/employee_dataset.csv')
    print('\n% Employee dataset', len(data), len(data[0]))
    return model, data

def travel_insurance():
    attrs = ['Agency', 'Agency Type', 'Distribution Channel', 'Product Name',
             'Duration', 'Destination', 'Net sales', 'Commision (in value)', 'Gender', 'Age']
    nums = ['Duration', 'Net Sales', 'Commision (in value)', 'Age']
    model = Classifier(attrs=attrs, numeric=nums, label='Claim', pos='No')
    data = model.load_data('data/travel_insurance/travel_insurance_dataset.csv')
    print('\n% Travel Insurance dataset', len(data), len(data[0]))
    return model, data

def telecom_churn():
    attrs = ['AccountWeeks', 'ContractRenewal', 'DataPlan', 'DataUsage',
             'CustServCalls', 'DayMins', 'DayCalls', 'MonthlyCharge', 'OverageFee']
    nums = attrs
    model = Classifier(attrs=attrs, numeric=nums, label='Churn', pos='0')
    data = model.load_data('data/telecom_churn/telecom_churn_dataset.csv')
    print('\n% Telecom Churn dataset', len(data), len(data[0]))
    return model, data

def customer_segmentation():
    attrs = ['Gender', 'Ever_Married', 'Age', 'Graduated', 'Profession',
             'Work_Experience', 'Spending_Score', 'Family_Size']
    nums = ['Age', 'Work_Experience', 'Spending_Score', 'Family_Size']
    model = Classifier(attrs=attrs, numeric=nums, label='Var_1', pos='Cat_6')
    data_train = model.load_data('data/customer_segmentation/train.csv')
    data_test = model.load_data('data/customer_segmentation/test.csv')
    print('\n% Customer Segmentation train dataset', len(data_train), len(data_train[0]))
    print('% Customer Segmentation test dataset', len(data_test), len(data_test[0]))
    return model, data_train, data_test

def water_quality():
    attrs = ['aluminium', 'ammonia', 'arsenic', 'barium', 'cadmium',
             'chloramine', 'chromium', 'copper','flouride', 'bacteria',
             'viruses','lead', 'nitrates', 'nitrites', 'mercury', 'perchlorate',
             'radium', 'selenium', 'silver', 'uranium']
    nums = attrs
    model = Classifier(attrs=attrs, numeric=nums, label='is_safe', pos='0')
    data = model.load_data('data/water_quality/water_quality_dataset.csv')
    print('\n% Water Quality dataset', len(data), len(data[0]))
    return model, data


def diabetes():
    attrs = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
             'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    nums = attrs
    model = Classifier(attrs=attrs, numeric=nums, label='Outcome', pos='0')
    data = model.load_data('data/diabetes/diabetes_dataset.csv')
    print('\n% Diabetes dataset', len(data), len(data[0]))
    return model, data

def stress():
    attrs = ['sr', 'rr', 't', 'lm', 'bo', 'rem', 'sh', 'hr']
    nums = attrs
    model = Classifier(attrs=attrs, numeric=nums, label='sl', pos='0')
    data = model.load_data('data/stress/stress_dataset.csv')
    print('\n% Stress dataset', len(data), len(data[0]))
    return model, data




    
