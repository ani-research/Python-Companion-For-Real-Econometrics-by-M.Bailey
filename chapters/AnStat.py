import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor



# Variance Inflation Factor


def give_vif(ols, drop_const=True):
    """
    Calculate Variance Inflation Factors (VIF) for an OLS model.

    Parameters
    ----------
    ols : statsmodels RegressionResults
        Fitted OLS model.
    drop_const : bool, default=True
        Whether to remove the intercept from the output.

    Returns
    -------
    pandas.DataFrame
    """

    X = ols.model.exog
    names = ols.model.exog_names

    vif = pd.DataFrame({
        "Variable": names,
        "VIF": [
            variance_inflation_factor(X, i)
            for i in range(X.shape[1])
        ]
    })

    if drop_const:
        vif = vif[vif["Variable"] != "Intercept"]

    return vif.reset_index(drop=True)



# Coefficient Plot


def coefficient_plot(
    ols,
    figsize=(6, 4),
    color="#0060ac",
    title="Regression coefficients",
    show_zero=True,
):
    """
    Plot regression coefficients with confidence intervals.
    """

    coef = ols.params.iloc[1:]
    ci = ols.conf_int().iloc[1:]

    y = np.arange(len(coef))

    plt.figure(figsize=figsize)

    plt.errorbar(
        coef,
        y,
        xerr=[
            coef - ci[0],
            ci[1] - coef,
        ],
        fmt="o",
        capsize=3,
        color=color,
    )

    if show_zero:
        plt.axvline(0, color="gray", linestyle="--")

    plt.yticks(y, coef.index)

    plt.xlabel("Estimated coefficient")
    plt.title(title)

    plt.tight_layout()
    plt.show()



# Group Effects (Interaction Models)


def group_effects(model, dummy, precision=4):
    """
    Return group-specific coefficients from an interaction model.

    Example
    -------
    wage ~ height81*C(male)

    gives coefficients for

    - Reference group
    - Dummy group

    including correct standard errors,
    t-statistics,
    p-values,
    and confidence intervals.
    """

    params = model.params.index

    main_effects = [
        p
        for p in params
        if ":" not in p
        and p != "Intercept"
        and not p.startswith(f"C({dummy})")
    ]

    rows = []

    # --------------------------------------------------
    # Reference group
    # --------------------------------------------------

    rows.append({
        "Group": "Reference",
        "Variable": "Intercept",
        "Coef": model.params["Intercept"],
        "Std.Err": model.bse["Intercept"],
        "t": model.tvalues["Intercept"],
        "P>|t|": model.pvalues["Intercept"],
        "CI Lower": model.conf_int().loc["Intercept", 0],
        "CI Upper": model.conf_int().loc["Intercept", 1],
    })

    for var in main_effects:

        rows.append({
            "Group": "Reference",
            "Variable": var,
            "Coef": model.params[var],
            "Std.Err": model.bse[var],
            "t": model.tvalues[var],
            "P>|t|": model.pvalues[var],
            "CI Lower": model.conf_int().loc[var, 0],
            "CI Upper": model.conf_int().loc[var, 1],
        })

    # --------------------------------------------------
    # Dummy group
    # --------------------------------------------------

    dummy_term = f"C({dummy})[T.1.0]"

    if dummy_term in params:

        tt = model.t_test(f"Intercept + {dummy_term} = 0")

        rows.append({
            "Group": dummy.capitalize(),
            "Variable": "Intercept",
            "Coef": tt.effect.item(),
            "Std.Err": tt.sd.item(),
            "t": tt.tvalue.item(),
            "P>|t|": tt.pvalue.item(),
            "CI Lower": tt.conf_int()[0, 0],
            "CI Upper": tt.conf_int()[0, 1],
        })

    for var in main_effects:

        interaction = f"{var}:C({dummy})[T.1.0]"

        if interaction in params:
            tt = model.t_test(f"{var} + {interaction} = 0")
        else:
            tt = model.t_test(f"{var} = 0")

        rows.append({
            "Group": dummy.capitalize(),
            "Variable": var,
            "Coef": tt.effect.item(),
            "Std.Err": tt.sd.item(),
            "t": tt.tvalue.item(),
            "P>|t|": tt.pvalue.item(),
            "CI Lower": tt.conf_int()[0, 0],
            "CI Upper": tt.conf_int()[0, 1],
        })

    df = pd.DataFrame(rows)

    numeric = [
        "Coef",
        "Std.Err",
        "t",
        "P>|t|",
        "CI Lower",
        "CI Upper",
    ]

    df[numeric] = df[numeric].round(precision)

    return df