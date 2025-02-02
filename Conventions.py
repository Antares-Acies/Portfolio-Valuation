import datetime
import logging
import numpy as np
import pandas as pd

def IsLeapYear(year):
    if year % 4 > 0:
        IsLeapYear = False
    elif year % 100 > 0:
        IsLeapYear = True
    elif year % 400 == 0:
        IsLeapYear = True
    else:
        IsLeapYear = False
    return IsLeapYear


def IsEndOfMonth(day, month, year):
    if month in [1, 3, 5, 7, 8, 10, 12]:
        IsEndOfMonth = day == 31
    if month in [4, 6, 9, 11]:
        IsEndOfMonth = day == 30
    if month in [2]:
        if IsLeapYear(year):
            IsEndOfMonth = day == 29
        else:
            IsEndOfMonth = day == 28
    return IsEndOfMonth


def Days360(StartYear, EndYear, StartMonth, EndMonth, StartDay, EndDay):
    Days360 = ((EndYear - StartYear) * 360) + ((EndMonth - StartMonth) * 30) + (EndDay - StartDay)
    return Days360


def TmpDays360Nasd(StartDate, EndDate, Method, UseEom):
    StartDay = StartDate.day
    StartMonth = StartDate.month
    StartYear = StartDate.year
    EndDay = EndDate.day
    EndMonth = EndDate.month
    EndYear = EndDate.year
    if (EndMonth == 2 and IsEndOfMonth(EndDay, EndMonth, EndYear)) and (
        (StartMonth == 2 and IsEndOfMonth(StartDay, StartMonth, StartYear)) or Method == 3
    ):
        EndDay = 30
    if EndDay == 31 and (StartDay >= 30 or Method == 3):
        EndDay = 30
    if StartDay == 31:
        StartDay = 30
    if UseEom and StartMonth == 2 and IsEndOfMonth(StartDay, StartMonth, StartYear):
        StartDay = 30
    return Days360(StartYear, EndYear, StartMonth, EndMonth, StartDay, EndDay)


def TmpDays360Euro(StartDate, EndDate):
    StartDay = StartDate.day
    StartMonth = StartDate.month
    StartYear = StartDate.year
    EndDay = EndDate.day
    EndMonth = EndDate.month
    EndYear = EndDate.year
    if StartDay == 31:
        StartDay = 30
    if EndDay == 31:
        EndDay = 30
    return Days360(StartYear, EndYear, StartMonth, EndMonth, StartDay, EndDay)


def DateDiff(StartDate, EndDate):
    return abs((StartDate - EndDate).days)


def TmpDiffDates(StartDate, EndDate, Basis):
    if Basis in [0]:
        TmpDiffDates = TmpDays360Nasd(StartDate, EndDate, 0, True)
    elif Basis in [1, 2, 3]:
        TmpDiffDates = DateDiff(StartDate, EndDate)
    elif Basis in [4]:
        TmpDiffDates = TmpDays360Euro(StartDate, EndDate)
    return TmpDiffDates


def TmpCalcAnnualBasis(StartDate, EndDate, Basis):
    if Basis in [0, 2, 4]:
        TmpCalcAnnualBasis = 360
    elif Basis in [3]:
        TmpCalcAnnualBasis = 365
    elif Basis in [1]:
        StartDay = StartDate.day
        StartMonth = StartDate.month
        StartYear = StartDate.year
        EndDay = EndDate.day
        EndMonth = EndDate.month
        EndYear = EndDate.year

        if StartYear == EndYear:
            if IsLeapYear(StartYear):
                TmpCalcAnnualBasis = 366
            else:
                TmpCalcAnnualBasis = 365
        elif (EndYear - 1) == StartYear and (
            StartMonth > EndMonth or (StartMonth == EndMonth and StartDay >= EndDay)
        ):
            if IsLeapYear(StartYear):
                if StartMonth < 2 or (StartMonth == 2 and StartDay <= 29):
                    TmpCalcAnnualBasis = 366
                else:
                    TmpCalcAnnualBasis = 365
            elif IsLeapYear(EndYear):
                if EndMonth > 2 or (EndMonth == 2 and EndDay == 29):
                    TmpCalcAnnualBasis = 366
                else:
                    TmpCalcAnnualBasis = 365
            else:
                TmpCalcAnnualBasis = 365
        else:
            TmpCalcAnnualBasis = 0
            for iYear in range(StartYear, EndYear + 1):
                if IsLeapYear(iYear):
                    TmpCalcAnnualBasis = TmpCalcAnnualBasis + 366
                else:
                    TmpCalcAnnualBasis = TmpCalcAnnualBasis + 365
            TmpCalcAnnualBasis = TmpCalcAnnualBasis / (EndYear - StartYear + 1)
    return TmpCalcAnnualBasis


def YearFrac(StartDate, EndDate, Basis):
    Numerator = TmpDiffDates(StartDate, EndDate, Basis)
    Denom = TmpCalcAnnualBasis(StartDate, EndDate, Basis)
    YearFrac = Numerator / Denom
    return YearFrac


class Conventions:
    def A_day_count(
        self,
        Payment_start_date,
        Payment_end_date,
        Convention,
        Maturity_date="",
        couponfreq=1,
        Next_coupon_date="",
        holiday_list=[],
        custom_daycount_conventions=None,
    ):
        Payment_start_date = pd.to_datetime(Payment_start_date)
        Payment_end_date = pd.to_datetime(Payment_end_date)

        years_start = Payment_start_date.year
        years_end = Payment_end_date.year
        months_start = Payment_start_date.month
        months_end = Payment_end_date.month
        days_start = Payment_start_date.day
        days_end = Payment_end_date.day

        # 30/360 Bond Basis
        if Convention == 1:
            if days_end == 31 and (days_start == 30 or days_start == 31):
                x = 30
            else:
                x = days_end
            if days_start == 31:
                y = 30
            else:
                y = days_start
            return ((years_end - years_start) * 360 + (months_end - months_start) * 30 + (x - y)) / 360

        # 30/360 US
        elif Convention == 2:
            if (
                (Payment_start_date + 1).astype("datetime64[M]").astype(int) % 12 + 1 == 3
                and ((Payment_start_date + 1).astype("datetime64[M]").astype(int) + 1 == 1)
                and ((Payment_end_date + 1).astype(int) % 12 + 1 == 3)
                and ((Payment_end_date + 1).astype("datetime64[M]").astype(int) + 1 == 1)
            ):
                x = 30
            elif days_end == 31 and (days_start == 30 or days_start == 31):
                x = 30
            else:
                x = days_end

            if ((Payment_start_date + 1).astype(int) % 12 + 1 == 3) and (
                (Payment_start_date + 1).astype("datetime64[M]").astype(int) + 1 == 1
            ):
                y = 30
            elif days_start == 31:
                y = 30
            else:
                y = days_start
            return ((years_end - years_start) * 360 + (months_end - months_start) * 30 + (x - y)) / 360

        # 30E/360
        elif Convention == 3:
            # Europen convention consisder each month to be of 30 day's irrespective of number of day's in a month (28/29/30/31)
            x=30
            y=30
            return ((years_end - years_start) * 360 + (months_end - months_start) * 30 + (x - y)) / 360

        # Act/360
        elif Convention == 6:
            days = (Payment_end_date - Payment_start_date).days
            return (days)/360

        # Act/365
        elif Convention == 7:
            days = (Payment_end_date - Payment_start_date).days
            return (days)/365

        # Act/Act
        elif Convention == 14:
            return YearFrac(
                pd.to_datetime(Payment_start_date),
                pd.to_datetime(Payment_end_date),
                1,
            )

        else:
            custom_daycount_conventions_filtered = custom_daycount_conventions.loc[
                custom_daycount_conventions["convention_name"] == Convention[0]
            ]
            numerator = custom_daycount_conventions_filtered["numerator"].iloc[0]
            if numerator in ["Actual", "ACTUAL", "ACT", "Act"]:
                numerator = (Payment_end_date - Payment_start_date).astype("int")
            elif str(numerator) == "30":
                if days_end == 31 and (days_start == 30 or days_start == 31):
                    x = 30
                else:
                    x = days_end
                if days_start == 31:
                    y = 30
                else:
                    y = days_start
                numerator = (years_end - years_start) * 360 + (months_end - months_start) * 30 + (x - y)

            numerator_adjustment = (
                custom_daycount_conventions_filtered["numerator_adjustment"].fillna(0).iloc[0]
            ).astype(float)
            numerator = numerator + numerator_adjustment
            denominator = custom_daycount_conventions_filtered["denominator"].iloc[0]
            if denominator in ["Actual", "ACTUAL", "ACT", "Act"]:
                denominator = 365.25
            else:
                denominator = float(denominator)
            denominator_adjustment = (
                custom_daycount_conventions_filtered["denominator_adjustment"].fillna(0).iloc[0]
            ).astype(float)
            denominator = denominator + denominator_adjustment
            return numerator / denominator

    def D_day_count(
        self,
        Val_Date,
        Coupon_Date,
        Convention,
        Maturity_date="none",
        couponfreq=1,
        Next_coupon_date_list=[],
        holiday_list=[],
        custom_daycount_conventions=None,
    ):
        days_Val = (Val_Date - Val_Date.astype("datetime64[M]")).astype(int) + 1
        days_Coupon = (Coupon_Date - Coupon_Date.astype("datetime64[M]")).astype(int) + 1
        years_Val = Val_Date.astype("datetime64[Y]").astype(int) + 1970
        years_Coupon = Coupon_Date.astype("datetime64[Y]").astype(int) + 1970
        months_Val = Val_Date.astype("datetime64[M]").astype(int) % 12 + 1
        months_Coupon = Coupon_Date.astype("datetime64[M]").astype(int) % 12 + 1

        # 30/360 Bond Basis
        if Convention == 1:
            a = np.where(days_Coupon == 31, 1, 0)
            b = np.where(days_Val == 30, 1, 0)
            c = np.where(days_Val == 31, 1, 0)
            ans = a & (b | c)
            x = np.where(ans == 1, 30, days_Coupon)
            y = np.where(days_Val == 31, 30, days_Val)
            day_count_list = (
                (years_Coupon - years_Val) * 360 + (months_Coupon - months_Val) * 30 + (x - y)
            ) / 360
            day_count = np.where(day_count_list <= 0, 0, day_count_list)
            return day_count

        # 30/360 US
        elif Convention == 2:
            a = np.where(months_Val == 2, 1, 0)
            b = np.where(days_Val == 1, 1, 0)
            c = np.where(months_Coupon == 2, 1, 0)
            d = np.where(days_Coupon == 1, 1, 0)
            ans = a & (b & c) & d
            x = np.where(ans == 1, 30, days_Coupon)
            e = np.where(months_Val == 2, 1, 0)
            f = np.where(days_Val == 1, 1, 0)
            g = np.where(days_Val == 31, 1, 0)
            ans_1 = e & (f | g)
            y = np.where(ans_1 == 1, 30, days_Val)
            day_count_list = (
                (years_Coupon - years_Val) * 360 + (months_Coupon - months_Val) * 30 + (x - y)
            ) / 360
            day_count = np.where(day_count_list <= 0, 0, day_count_list)
            return day_count

        # 30E/360
        elif Convention == 3:
            x = np.where(days_Coupon == 31, 30, days_Coupon)
            y = np.where(days_Val == 31, 30, days_Val)
            day_count_list = (
                (years_Coupon - years_Val) * 360 + (months_Coupon - months_Val) * 30 + (x - y)
            ) / 360
            day_count = np.where(day_count_list <= 0, 0, day_count_list)
            return day_count

        # Act/360
        elif Convention == 6:
            day_count_list = ((Coupon_Date - Val_Date).astype("float")) / 360
            day_count = np.where(day_count_list <= 0, 0, day_count_list)
            return day_count

        # Act/365
        elif Convention == 7:
            day_count_list = ((Coupon_Date - Val_Date).astype("float")) / 365
            day_count = np.where(day_count_list <= 0, 0, day_count_list)
            return day_count

        # Act/Act
        elif Convention == 14:

            def vectorized_yearfrac_calc(Coupon_Date, Val_Date):
                return YearFrac(
                    pd.to_datetime(Val_Date), pd.to_datetime(Coupon_Date), 1
                )

            yearfrac_vector = np.vectorize(vectorized_yearfrac_calc)
            day_count_list = yearfrac_vector(Coupon_Date, Val_Date)
            day_count = np.where(day_count_list <= 0, 0, day_count_list)
            return day_count

        else:
            custom_daycount_conventions_filtered = custom_daycount_conventions.loc[
                custom_daycount_conventions["convention_name"] == Convention[0]
            ]
            numerator = custom_daycount_conventions_filtered["numerator"].iloc[0]
            if numerator in ["Actual", "ACTUAL", "ACT", "Act"]:
                numerator = (Coupon_Date - Val_Date).astype("float")
            elif str(numerator) == "30":
                a = np.where(days_Coupon == 31, 1, 0)
                b = np.where(days_Val == 30, 1, 0)
                c = np.where(days_Val == 31, 1, 0)
                ans = a & (b | c)
                x = np.where(ans == 1, 30, days_Coupon)
                y = np.where(days_Val == 31, 30, days_Val)
                numerator = (years_Coupon - years_Val) * 360 + (months_Coupon - months_Val) * 30 + (x - y)

            numerator_adjustment = (
                custom_daycount_conventions_filtered["numerator_adjustment"].fillna(0).iloc[0]
            ).astype(float)
            numerator = numerator + numerator_adjustment
            denominator = custom_daycount_conventions_filtered["denominator"].iloc[0]
            if denominator in ["Actual", "ACTUAL", "ACT", "Act"]:
                denominator = 365.25
            else:
                denominator = float(denominator)
            denominator_adjustment = (
                custom_daycount_conventions_filtered["denominator_adjustment"].fillna(0).iloc[0]
            ).astype(float)
            denominator = denominator + denominator_adjustment
            day_count_list = numerator / denominator
            day_count = np.where(day_count_list <= 0, 0, day_count_list)
            return day_count

    def business_day(self, date, convention, holiday_list, business_days="1111100"):
        # Following
        if convention == 1:
            i = 0
            while np.busday_count(date, date + i, business_days, holiday_list) == 0:
                if np.busday_count(date, date + i + 1, business_days, holiday_list) != 0:
                    return date + i
                i += 1
        # Preceding
        if convention == 2:
            if np.busday_count(date, date + 1, business_days, holiday_list) != 0:
                return date
            else:
                i = 0
                while np.busday_count(date - i, date + 1, business_days, holiday_list) == 0:
                    if np.busday_count(date - (i + 1), date + 1, business_days, holiday_list) != 0:
                        return date - (i + 1)
                    i += 1

        # Modified Following
        if convention == 3:
            i = 0
            while np.busday_count(date, date + i, business_days, holiday_list) == 0:        
                if np.busday_count(date, date + (i + 1), business_days, holiday_list) != 0:

                    current_date = date
                    next_date = date + 1
                    current_date_month_precision =  current_date.astype('datetime64[M]').astype('int64') % 12 + 1
                    next_date_month_precision = next_date.astype('datetime64[M]').astype('int64') % 12 + 1

                    if current_date_month_precision == next_date_month_precision :
                        return date + i
                    else:
                        if np.busday_count(date, date + 1, business_days, holiday_list) != 0:
                            return date
                        else:
                            j = 0
                            while np.busday_count(date - j, date + 1, business_days, holiday_list) == 0:
                                if (
                                    np.busday_count(date - (j + 1), date + 1, business_days, holiday_list)
                                    != 0
                                ):
                                    return date - (j + 1)
                                j += 1
                i += 1

        # No adjustment
        if convention in [5, "EOM"]:
            return date

    def eomonth(self, d, maturity_date="", months=0):
        maturity_date = pd.to_datetime(maturity_date).date()
        if isinstance(d, pd.core.indexes.datetimes.DatetimeIndex):
            d = pd.to_datetime(d).date
        else:
            pass

        months = int(months)
        if isinstance(d, list):
            return [self.eomonth(d0, maturity_date, months) for d0 in d]
        elif isinstance(d, pd.Series):
            return d.map(lambda d0: self.eomonth(d0, maturity_date, months))
        elif isinstance(d, np.ndarray):
            function = np.vectorize(self.eomonth)
            return np.array(function(d, maturity_date)).astype("datetime64[D]")
        else:
            if maturity_date != "":
                if maturity_date == d:
                    return d
            d = pd.to_datetime(d)
            y, m = divmod(d.month + months + 1, 12)
            if m == 0:
                y -= 1
                m = 12
            final_d = datetime.datetime(d.year + y, m, 1) - datetime.timedelta(days=1)
            final_d = pd.to_datetime(final_d).date()

            if final_d>maturity_date:
                return np.datetime64(maturity_date)
            else:
                return final_d


conventions = Conventions()
