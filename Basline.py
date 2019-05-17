import pandas as pd
import numpy as np
import datetime

def _read_data():
    train_planned_prod = pd.read_csv('TrainPlannedDailyProductionP.csv')
    train_actual_prod = pd.read_csv('TrainProdDataP.csv')
    train_cons = pd.read_csv('TrainActualConsumptionDataP.csv')
    test_actual_prod = pd.read_csv('TestProdDataP.csv')
    test_cons = pd.read_csv('TestActualConsumptionDataP.csv')

    return train_planned_prod, train_actual_prod, train_cons, test_actual_prod, test_cons


def _calculate_mape(Y_real, Y_pred):
    return np.sum(np.abs(Y_real - Y_pred)) / np.sum(Y_pred)


train_planned_prod, train_actual_prod, train_cons, test_actual_prod, test_cons = _read_data()

cons = pd.concat([train_cons, test_cons], ignore_index=True)

cons['ConsumptionDate'] = pd.to_datetime(cons['ConsumptionDate'], format='%d/%m/%Y %H:%M')
cons['ConsumptionDate'] = cons['ConsumptionDate'] + datetime.timedelta(days=1)
test_cons['ConsumptionDate'] = pd.to_datetime(test_cons['ConsumptionDate'], format='%d/%m/%Y %H:%M')

cons['predict_cons'] = cons['ActualConsumption'].astype('float64')
test = test_cons.merge(cons, how='inner', on='ConsumptionDate')
print(test['predict_cons'].dtype)
print(_calculate_mape(test['ActualConsumption_x'], test['predict_cons']))
