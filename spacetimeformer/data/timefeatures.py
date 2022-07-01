import pandas as pd


def time_features(
    dates,
    main_df=None,
    use_features=["year", "month", "day", "weekday", "hour", "minute"],
    time_col_name="Datetime",
):
    if main_df is None:
        main_df = pd.DataFrame({})
    else:
        main_df = main_df.copy()
    years = dates.apply(lambda row: row.year)
    max_year = years.max()
    min_year = years.min()

    if "year" in use_features:
        main_df["Year"] = dates.apply(
            lambda row: (row.year - min_year) / max(1.0, (max_year - min_year))
        )

    if "month" in use_features:
        main_df["Month"] = dates.apply(
            lambda row: 2.0 * ((row.month - 1) / 11.0) - 1.0, 1
        )
    if "day" in use_features:
        main_df["Day"] = dates.apply(lambda row: 2.0 * ((row.day - 1) / 30.0) - 1.0, 1)
    if "weekday" in use_features:
        main_df["Weekday"] = dates.apply(
            lambda row: 2.0 * (row.weekday() / 6.0) - 1.0, 1
        )
    if "hour" in use_features:
        main_df["Hour"] = dates.apply(lambda row: 2.0 * ((row.hour) / 23.0) - 1.0, 1)
    if "minute" in use_features:
        main_df["Minute"] = dates.apply(
            lambda row: 2.0 * ((row.minute) / 59.0) - 1.0, 1
        )

    main_df[time_col_name] = dates
    return main_df
