from sklearn.metrics import mean_absolute_error
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as usedModel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def to_ordinal(x, frmt):
    try:
        return pd.to_datetime(x, format=frmt).toordinal()
    except:
        return 0


# load training and test data
train_data = pd.read_csv(
    'fixtures/train.csv',
    converters={
        'sale_date': lambda x: to_ordinal(x, '%Y-%m-%d'),
        'refitted_date': lambda x: to_ordinal(x, '%Y/%m'),
        'created_date': lambda x: to_ordinal(x, '%Y/%m')
    }
)
test_data = pd.read_csv(
    'fixtures/test.csv',
    converters={
        'sale_date': lambda x: to_ordinal(x, '%Y-%m-%d'),
        'refitted_date': lambda x: to_ordinal(x, '%Y/%m'),
        'created_date': lambda x: to_ordinal(x, '%Y/%m')
    }
)

# choose features and price as training target
featuresGroups = [
    # ['cabins', 'decks', 'bathrooms'],
    # ['sale_date', 'cabins', 'decks', 'bathrooms'],
    # ['producer', 'total_area'],
    # ['producer', 'total_area', 'refitted_date', 'created_date'],
    # ['crew_area','cabins_area','maintenance_area','crew_area_coef'],
    # ['total_area_coef','weight_distribution_x','weight_distribution_y','engine_thrust'],
    # ['radar', 'bow_thruster', 'autopilot', 'solar_panels' ],
    # ['navi_quality', 'state', 'overall_quality'],
    # # without dates
    # [
    #     'cabins', 'decks', 'bathrooms', 'producer', 'total_area', 'crew_area','cabins_area','maintenance_area','crew_area_coef',
    #     'total_area_coef','weight_distribution_x','weight_distribution_y','engine_thrust', 'radar', 'bow_thruster', 'autopilot', 'solar_panels',
    #     'navi_quality', 'state', 'overall_quality'
    # ],
    # # with dates
    [
        'cabins', 'decks', 'bathrooms', 'producer', 'total_area', 'crew_area', 'cabins_area', 'maintenance_area',
        'crew_area_coef',
        'total_area_coef', 'weight_distribution_x', 'weight_distribution_y', 'engine_thrust', 'radar', 'bow_thruster',
        'autopilot', 'solar_panels',
        'navi_quality', 'state', 'overall_quality', 'sale_date', 'refitted_date', 'created_date'
    ],
    # ['cabins'],
    # ['decks'],
    # ['bathrooms'],
    # ['producer'],
    # ['total_area'],
    # ['crew_area'],
    # ['cabins_area'],
    # ['maintenance_area'],
    # ['crew_area_coef'],
    # ['total_area_coef'],
    # ['weight_distribution_x'],
    # ['weight_distribution_y'],
    # ['engine_thrust'],
    # ['radar'],
    # ['bow_thruster'],
    # ['autopilot'],
    # ['solar_panels'],
    # ['navi_quality'],
    # ['state'],
    # ['overall_quality'],
    # ['sale_date'],
    # ['refitted_date'],
    # ['created_date']
    ['crew_area', 'overall_quality', 'cabins_area', 'crew_area_coef', 'bathrooms', 'navi_quality', 'maintenance_area'],
    # ['crew_area', 'overall_quality', 'cabins_area', 'crew_area_coef', 'bathrooms', 'navi_quality'],
    # ['crew_area', 'overall_quality', 'cabins_area', 'crew_area_coef', 'bathrooms'],
    ['crew_area', 'overall_quality', 'cabins_area', 'crew_area_coef', 'navi_quality'],
    # ['crew_area', 'overall_quality', 'cabins_area', 'crew_area_coef'],
    # ['crew_area', 'overall_quality', 'cabins_area'],
    # ['crew_area', 'overall_quality'],
    # ['crew_area'],
]

for i, features in enumerate(featuresGroups):
    train_x = train_data[features]
    train_y = train_data['price']

    # create model
    # model = usedModel()
    model = make_pipeline(StandardScaler(), usedModel(criterion='absolute_error'))
    model.fit(train_x, train_y)

    # predict
    test_x = test_data[features]
    test_data['price'] = model.predict(test_x)

    test_data[['id', 'price']].to_csv("output/%s-my_submission.csv" % (str(i + 1)), index=False)

    # check
    checkData = model.predict(train_x)
    print(str(i + 1) + ' - ' + str(mean_absolute_error(train_y, checkData)) + ' - ' + str(model.score(train_x, train_y)))
