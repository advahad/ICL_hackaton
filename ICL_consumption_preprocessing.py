import pandas as pd
import numpy as np


def _read_data():
    train_planned_prod = pd.read_csv('TrainPlannedDailyProduction.csv')
    train_actual_prod = pd.read_csv('TrainProdData.csv')
    train_cons = pd.read_csv('TrainActualConsumptionData.csv')
    test_actual_prod = pd.read_csv('TestProdData.csv')
    test_cons = pd.read_csv('TestActualConsumptionData.csv')

    return train_planned_prod, train_actual_prod, train_cons, test_actual_prod, test_cons



def _write_csv():
    train_planned_prod.to_csv('TrainPlannedDailyProductionP.csv', index=False)
    train_actual_prod.to_csv('TrainProdDataP.csv', index=False)
    train_cons.to_csv('TrainActualConsumptionDataP.csv', index=False)
    test_actual_prod.to_csv('TestProdDataP.csv', index=False)
    test_cons.to_csv('TestActualConsumptionDataP.csv', index=False)


def create_regression_data(prod, cons, e):
    data_set = prod[['ProductionDate', 'temp', 'humidity', 'barometric']]
    data_set['cons/prod'] = ((cons['P1Cons'] / (prod['P1prod'] + e)) + (cons['P2Cons'] / (prod['P2prod'] + e)) +\
                               (cons['P3Cons'] / (prod['P3prod'] + e)) + (cons['P4Cons']/(prod['P4prod'] + e)) +\
                               (cons['P5Cons'] / (prod['P5prod'] + e)) + (cons['P6Cons1'] / (prod['P6prod'] + e))) / 6
    data_set.to_csv('weather_regression.csv')


def _missing_values(df):
    for column in df.columns[1:]:
        df[column] = pd.to_numeric(df[column], errors='coerce')
        df.interpolate(method='linear', axis=0, limit_direction='both', inplace=True)
    return df


def actual_consumption(cons):
    cons['gen_sum'] = 0.03 * cons['Gen1num1'] + 0.03 * cons['Gen1num2'] + 0.06 * cons['Gen2']
    col_list = list(cons.columns[2:])
    print(col_list)

    col_list.remove('Gen1num1')
    col_list.remove('Gen1num2')
    col_list.remove('Gen2')

    cons['residual'] = cons['ActualConsumption'] - cons[col_list].sum(axis=1)
    # get highest valid value
    quantile = cons[cons['residual'] > 0]['residual'].quantile(0.90)
    residual = np.mean(cons[cons['residual'] < quantile]['residual'])

    cons = cons.drop('residual', axis=1)

    cons.loc[cons['ActualConsumption'] < 123, 'ActualConsumption']\
        = cons[cons['ActualConsumption'] < 123][col_list].sum(axis=1) + residual
    cons.loc[cons['ActualConsumption'] < 120, 'ActualConsumption'] \
        = 120
    return cons


def scaling(df):
    pass


train_planned_prod, train_actual_prod, train_cons, test_actual_prod, test_cons = _read_data()

# take care of missing values
train_planned_prod = _missing_values(train_planned_prod)
train_actual_prod = _missing_values(train_actual_prod)
train_cons = _missing_values(train_cons)
test_actual_prod = _missing_values(test_actual_prod)
test_cons = _missing_values(test_cons)

train_actual_prod['P6prod'] = train_actual_prod['P6prod'] / 5.9
test_actual_prod['P6prod'] = test_actual_prod['P6prod'] / 5.9

train_cons = actual_consumption(train_cons)
test_cons = actual_consumption(test_cons)
create_regression_data(train_actual_prod, train_cons, 0.0001)
train_actual_prod.set_index(train_actual_prod['ProductionDate'])
p1 = train_actual_prod.filter(regex=('P1'))

_write_csv()
