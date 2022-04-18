import holidays
import pandas as pd


def is_business_day(date):
    return bool(len(pd.bdate_range(date, date)))


def is_holiday(date, country="ru"):
    assert country in ("ru", )

    return date in holidays.RU()
